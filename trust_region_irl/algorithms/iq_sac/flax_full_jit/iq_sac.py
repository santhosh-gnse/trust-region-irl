import os
import shutil
import json
from copy import deepcopy
import logging
import time
import tree
import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from trust_region_irl.algorithms.iq_sac.flax_full_jit.general_properties import GeneralProperties
from trust_region_irl.algorithms.iq_sac.flax_full_jit.policy import get_policy
from trust_region_irl.algorithms.iq_sac.flax_full_jit.critic import get_critic
from trust_region_irl.algorithms.iq_sac.flax_full_jit.entropy_coefficient import EntropyCoefficient, ConstantEntropyCoefficient
from trust_region_irl.algorithms.iq_sac.flax_full_jit.rl_train_state import RLTrainState
from trust_region_irl.algorithms.data_utils import prepare_expert_data, expert_data_spec

rlx_logger = logging.getLogger("rl_x")


class IQ_SAC:
    def __init__(self, config, train_env, eval_env, run_path, writer):
        self.config = config
        self.train_env = train_env
        self.eval_env = eval_env
        self.writer = writer

        self.save_model = config.runner.save_model
        self.save_path = os.path.join(run_path, "models")
        self.track_console = config.runner.track_console
        self.track_tb = config.runner.track_tb
        self.track_wandb = config.runner.track_wandb
        self.seed = config.environment.seed
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_parallel_seeds = config.algorithm.nr_parallel_seeds
        self.nr_envs = config.environment.nr_envs
        self.render = config.environment.render
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.buffer_size = config.algorithm.buffer_size
        self.learning_starts = config.algorithm.learning_starts
        self.batch_size = config.algorithm.batch_size
        self.tau = config.algorithm.tau
        self.gamma = config.algorithm.gamma
        self.target_entropy = config.algorithm.target_entropy
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_and_save_frequency = config.algorithm.evaluation_and_save_frequency
        self.evaluation_active = config.algorithm.evaluation_active
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.total_training_timesteps = self.total_timesteps - self.learning_starts
        if config.algorithm.evaluation_and_save_frequency == -1:
            self.evaluation_and_save_frequency = self.nr_envs * (self.total_training_timesteps // self.nr_envs)
        self.nr_eval_save_iterations = self.total_training_timesteps // self.evaluation_and_save_frequency
        self.nr_loggings_per_eval_save_iteration = self.evaluation_and_save_frequency // self.logging_frequency
        self.nr_updates_per_logging_iteration = self.logging_frequency // self.nr_envs
        self.horizon = self.train_env.horizon

        # IQ Learn Specific
        self.reg_mult = config.algorithm.reg_mult
        self.data_path = config.algorithm.data_path
        self.num_data_samples = np.load(self.data_path)["states"].shape[0]
        self.gp_lambda = config.algorithm.gp_lambda
        self.learn_ent_coeff = config.algorithm.learn_ent_coeff
        self.v0_loss = config.algorithm.v0_loss
        self.nr_q_updates_per_step = config.algorithm.nr_q_updates_per_step
        self.use_target_q = config.algorithm.use_target_q
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.use_lsiq = config.algorithm.use_lsiq
        self.Q_max = 1.0 / (self.reg_mult * (1 - self.gamma))
        self.Q_min = - 1.0 / (self.reg_mult * (1 - self.gamma))

        if self.nr_parallel_seeds > 1:
            raise ValueError("Parallel seeds are not supported yet. This is mainly limited by not being able to log mutliple wandb runs at the same time.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")
        
        self.key = jax.random.PRNGKey(self.seed)
        self.key, reset_key, policy_key, critic_key, entropy_coefficient_key = jax.random.split(self.key, 5)
        reset_key = jax.random.split(reset_key, 1)

        self.env_as_low = self.train_env.single_action_space.low
        self.env_as_high = self.train_env.single_action_space.high
        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape

        self.policy, self.get_processed_action = get_policy(config, self.train_env)
        self.critic = get_critic(config, self.train_env)
        
        if self.target_entropy == "auto":
            self.target_entropy = -np.prod(self.train_env.single_action_space.shape).item()
        else:
            self.target_entropy = float(self.target_entropy)

        if self.learn_ent_coeff:
            self.entropy_coefficient = EntropyCoefficient(config.algorithm.init_ent_coeff)
        else:
            self.entropy_coefficient = ConstantEntropyCoefficient(config.algorithm.init_ent_coeff)

        self.policy.apply = jax.jit(self.policy.apply)
        self.critic.apply = jax.jit(self.critic.apply)
        self.entropy_coefficient.apply = jax.jit(self.entropy_coefficient.apply)
        self.nr_q_updates = 0

        def linear_schedule(count):
            step = (count * self.nr_envs) - self.learning_starts
            total_steps = self.total_timesteps - self.learning_starts
            fraction = 1.0 - (step / total_steps)
            return self.learning_rate * fraction
        
        self.q_learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate
        self.policy_learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate
        self.entropy_learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

        env_state = self.train_env.reset(reset_key, False)
        self.dummy_state = env_state.next_observation
        self.dummy_action = jnp.array([self.train_env.single_action_space.sample(reset_key[0])])

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, self.dummy_state),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=self.policy_learning_rate),
            ),
        )

        self.critic_state = RLTrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, self.dummy_state, self.dummy_action),
            target_params=self.critic.init(critic_key, self.dummy_state, self.dummy_action),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=self.q_learning_rate),
            ),
        )

        self.entropy_coefficient_state = TrainState.create(
            apply_fn=self.entropy_coefficient.apply,
            params=self.entropy_coefficient.init(entropy_coefficient_key),
            tx=optax.inject_hyperparams(optax.adam)(learning_rate=self.entropy_learning_rate)
        )

        if self.save_model:
            os.makedirs(self.save_path)
            self.latest_model_file_name = "latest.model"
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        
    
    def train(self):
        def jitable_train_function(key, parallel_seed_id):
            key, reset_key = jax.random.split(key, 2)
            reset_keys = jax.random.split(reset_key, self.nr_envs)
            env_state = self.train_env.reset(reset_keys, False)

            # Expert demonstrations
            def _prepare_expert_data():
                return prepare_expert_data(self.data_path)
            
            demonstrations = jax.experimental.io_callback(_prepare_expert_data, expert_data_spec(num_samples=self.num_data_samples, state_dim=self.train_env.single_observation_space.shape[0], action_dim=self.train_env.single_action_space.shape[0]))
            expert_states = demonstrations["states"]
            expert_next_states = demonstrations["next_states"]
            expert_actions = demonstrations["actions"]
            expert_absorbing = demonstrations["absorbing"].flatten()

            policy_state = self.policy_state
            critic_state = self.critic_state
            entropy_coefficient_state = self.entropy_coefficient_state

            # Replay buffer
            capacity = int(self.buffer_size // self.nr_envs)
            states_buffer = jnp.zeros((capacity, self.nr_envs) + (self.dummy_state.shape[1],), dtype=jnp.float32)
            next_states_buffer = jnp.zeros((capacity, self.nr_envs) + (self.dummy_state.shape[1],), dtype=jnp.float32)
            actions_buffer = jnp.zeros((capacity, self.nr_envs) + (self.dummy_action.shape[1],), dtype=jnp.float32)
            rewards_buffer = jnp.zeros((capacity, self.nr_envs), dtype=jnp.float32)
            terminations_buffer = jnp.zeros((capacity, self.nr_envs), dtype=jnp.float32)
            replay_buffer = {
                "states": states_buffer,
                "next_states": next_states_buffer,
                "actions": actions_buffer,
                "rewards": rewards_buffer,
                "terminations": terminations_buffer,
                "pos": jnp.zeros((), dtype=jnp.int32),
                "size": jnp.zeros((), dtype=jnp.int32)
            }

            # Fill replay buffer until learning_starts
            prefill_iterations = int(np.ceil(self.learning_starts / self.nr_envs)) if self.learning_starts > 0 else 0
            if prefill_iterations > 0:
                def fill_replay_buffer(carry, _):
                    env_state, replay_buffer, key = carry
                    key, subkey = jax.random.split(key)
                    observation = env_state.next_observation
                    processed_action = self.train_env.single_action_space.sample(subkey)
                    action = (processed_action - self.env_as_low) / (self.env_as_high - self.env_as_low) * 2.0 - 1.0
                    env_state = self.train_env.step(env_state, processed_action)

                    replay_buffer["states"] = replay_buffer["states"].at[replay_buffer["pos"]].set(observation)
                    replay_buffer["next_states"] = replay_buffer["next_states"].at[replay_buffer["pos"]].set(env_state.actual_next_observation)
                    replay_buffer["actions"] = replay_buffer["actions"].at[replay_buffer["pos"]].set(action)
                    replay_buffer["rewards"] = replay_buffer["rewards"].at[replay_buffer["pos"]].set(env_state.reward)
                    replay_buffer["terminations"] = replay_buffer["terminations"].at[replay_buffer["pos"]].set(env_state.terminated)
                    replay_buffer["pos"] = (replay_buffer["pos"] + 1) % capacity
                    replay_buffer["size"] = jnp.minimum(replay_buffer["size"] + 1, capacity)

                    return (env_state, replay_buffer, key), None

                (env_state, replay_buffer, key), _ = jax.lax.scan(fill_replay_buffer, (env_state, replay_buffer, key), jnp.arange(prefill_iterations))


            def eval_save_iteration(eval_save_iteration_carry, eval_save_iteration_step):
                policy_state, critic_state, entropy_coefficient_state, replay_buffer, \
                (expert_states, expert_actions, expert_next_states, expert_absorbing), \
                env_state, key = eval_save_iteration_carry

                def logging_iteration(logging_iteration_carry, logging_iteration_step):
                    policy_state, critic_state, entropy_coefficient_state, replay_buffer, \
                    (expert_states, expert_actions, expert_next_states, expert_absorbing), \
                    env_state, key = logging_iteration_carry

                    def learning_iteration(learning_iteration_carry, learning_iteration_step):
                        policy_state, critic_state, entropy_coefficient_state, replay_buffer, \
                        (expert_states, expert_actions, expert_next_states, expert_absorbing), \
                        env_state, key = learning_iteration_carry

                        # Acting
                        key, subkey = jax.random.split(key)
                        observation = env_state.next_observation
                        dist = self.policy.apply(policy_state.params, observation)
                        action = dist.sample(seed=subkey)
                        processed_action = self.get_processed_action(action)
                        env_state = self.train_env.step(env_state, processed_action)

                        # Adding to replay buffer
                        replay_buffer["states"] = replay_buffer["states"].at[replay_buffer["pos"]].set(observation)
                        replay_buffer["next_states"] = replay_buffer["next_states"].at[replay_buffer["pos"]].set(env_state.actual_next_observation)
                        replay_buffer["actions"] = replay_buffer["actions"].at[replay_buffer["pos"]].set(action)
                        replay_buffer["rewards"] = replay_buffer["rewards"].at[replay_buffer["pos"]].set(env_state.reward)
                        replay_buffer["terminations"] = replay_buffer["terminations"].at[replay_buffer["pos"]].set(env_state.terminated)
                        replay_buffer["pos"] = (replay_buffer["pos"] + 1) % capacity
                        replay_buffer["size"] = jnp.minimum(replay_buffer["size"] + 1, capacity)

                        if self.render:
                            def render(env_state):
                                return self.train_env.render(env_state)
                            
                            env_state = jax.experimental.io_callback(render, env_state, env_state)

                        """ IQ Q-Function Update"""
                        def get_v(critic_params, policy_params, ent_params, states, key):
                            alpha_with_grad = self.entropy_coefficient.apply(ent_params)
                            alpha = stop_gradient(alpha_with_grad)
                            dist = self.policy.apply(policy_params, states)
                            a = dist.sample(seed=key)
                            logp = dist.log_prob(a)[..., None]
                            q = self.critic.apply(critic_params, states, a)
                            q = jnp.squeeze(q, axis=-1).min(axis=0)[:, None]
                            v = q - alpha * logp
                            return stop_gradient(v)

                        def regularizer_loss(absorbing, reward, gamma, reg_mult, treat_absorbing_states=False):
                            reg_abs = absorbing if treat_absorbing_states else jnp.zeros_like(absorbing)
                            chi2 = ((1.0 - reg_abs) * reg_mult * jnp.square(reward) +
                                    reg_abs * (1.0 - gamma) * reg_mult * jnp.square(reward)).mean()
                            return chi2

                        def gradient_penalty(critic_params, obs, act, num_rollout, key_gp):
                            def _gp(_):
                                obs_plcy, obs_exp = obs[:num_rollout], obs[num_rollout:]
                                act_plcy, act_exp = act[:num_rollout], act[num_rollout:]
                                alpha = jax.random.uniform(key_gp, shape=(num_rollout, 1))
                                while alpha.ndim < obs_exp.ndim:
                                    alpha = alpha[..., None]
                                s_interp = alpha * obs_exp + (1.0 - alpha) * obs_plcy
                                a_interp = alpha * act_exp + (1.0 - alpha) * act_plcy

                                def q_single(s, a):
                                    qv = self.critic.apply(critic_params, s[None, ...], a[None, ...])
                                    qv = jnp.squeeze(qv, axis=-1).min(axis=0)
                                    return qv[0]

                                grad_s, grad_a = jax.vmap(
                                    jax.grad(lambda ss, aa: q_single(ss, aa), argnums=(0, 1))
                                )(s_interp, a_interp)
                                grad_s_flat = grad_s.reshape((grad_s.shape[0], -1))
                                grad_a_flat = grad_a.reshape((grad_a.shape[0], -1))
                                grad_norm = jnp.linalg.norm(jnp.concatenate([grad_s_flat, grad_a_flat], axis=-1), axis=-1)
                                return self.gp_lambda * jnp.mean((grad_norm - 1.0) ** 2)

                            return jax.lax.cond(self.gp_lambda > 0.0, _gp, lambda _: 0.0, operand=None)

                        def iq_loss_fn(critic_params, policy_params, entropy_params,
                                       states, actions, next_states, terminations,
                                       expert_states, expert_actions, expert_next_states, expert_absorbing,
                                       key_demo, key_v_next, key_v, key_gp):
                            gamma = jnp.asarray(self.gamma, dtype=jnp.float32)
                            num_rollout = states.shape[0]

                            perm = jax.random.permutation(key_demo, expert_states.shape[0])
                            idx = perm[:num_rollout]
                            exp_s = expert_states[idx]
                            exp_a = expert_actions[idx]
                            exp_s_next = expert_next_states[idx]
                            exp_abs = expert_absorbing[idx].astype(jnp.float32)

                            obs = jnp.concatenate([states, exp_s], axis=0)
                            act = jnp.concatenate([actions, exp_a], axis=0)
                            next_obs = jnp.concatenate([next_states, exp_s_next], axis=0)

                            absorbing = jnp.concatenate([terminations.astype(jnp.float32), exp_abs], axis=0)[:, None]
                            is_expert = jnp.concatenate([jnp.zeros((num_rollout,), dtype=jnp.float32),
                                                         jnp.ones((num_rollout,), dtype=jnp.float32)], axis=0)
                            expert_denom = is_expert.sum() + 1e-8

                            q_values = self.critic.apply(critic_params, obs, act)
                            q_values = jnp.squeeze(q_values, axis=-1).min(axis=0)[:, None]

                            target_params = jax.lax.cond(
                                self.use_target_q,
                                lambda _: critic_state.target_params,
                                lambda _: critic_params,
                                operand=None
                            )
                            next_v = get_v(target_params, policy_params, entropy_params, next_obs, key_v_next)
                            if self.use_lsiq:
                                y = (1.0 - absorbing) * gamma * jnp.clip(next_v, self.Q_min, self.Q_max)
                            else:
                                y = (1.0 - absorbing) * gamma * next_v

                            reward = q_values - y
                            reward_flat = reward.squeeze(-1)

                            exp_reward_mean = (reward_flat * is_expert).sum() / expert_denom
                            if self.use_lsiq:
                                loss_term1 = jnp.mean(optax.losses.squared_error(
                                    q_values * is_expert[:, None], jnp.ones_like(q_values) * self.Q_max * is_expert[:, None]
                                ))
                            else:
                                loss_term1 = -exp_reward_mean

                            V = get_v(critic_params, policy_params, entropy_params, obs, key_v)
                            value = (V - y).squeeze(-1)

                            if self.v0_loss:
                                V_flat = V.squeeze(-1)
                                V_exp_mean = (V_flat * is_expert).sum() / expert_denom
                                loss_term2 = (1.0 - gamma) * V_exp_mean
                            else:
                                loss_term2 = value.mean()

                            chi2_loss = regularizer_loss(absorbing, reward, gamma, self.reg_mult, treat_absorbing_states=True)
                            loss_gp = gradient_penalty(critic_params, obs, act, num_rollout, key_gp)
                            loss_Q = loss_term1 + loss_term2 + chi2_loss + loss_gp

                            diff_exp = reward_flat - exp_reward_mean
                            exp_reward_var = ((diff_exp ** 2) * is_expert).sum() / expert_denom
                            exp_reward_std = jnp.sqrt(exp_reward_var)

                            metrics = {
                                "iq/loss_q": loss_Q,
                                "iq/loss_term1_expert": loss_term1,
                                "iq/loss_term2_value": loss_term2,
                                "iq/chi2_loss": chi2_loss,
                                "iq/gp_loss": loss_gp,
                                "iq/reward_mean": reward_flat.mean(),
                                "iq/reward_expert_mean": exp_reward_mean,
                                "iq/reward_expert_std": exp_reward_std,
                            }
                            return loss_Q, metrics

                        def iq_update_step(carry, _):
                            policy_state, critic_state, entropy_coefficient_state, replay_buffer, \
                            (expert_states, expert_actions, expert_next_states, expert_absorbing), key = carry
                            key, rb_key, demo_key, vnext_key, v_key, gp_key = jax.random.split(key, 6)

                            idx1 = jax.random.randint(rb_key, (self.batch_size,), 0, replay_buffer["size"])
                            idx2 = jax.random.randint(rb_key, (self.batch_size,), 0, self.nr_envs)
                            states_b = replay_buffer["states"][idx1, idx2]
                            next_states_b = replay_buffer["next_states"][idx1, idx2]
                            actions_b = replay_buffer["actions"][idx1, idx2]
                            terminations_b = replay_buffer["terminations"][idx1, idx2]

                            (loss_q, iq_metrics), critic_grads = jax.value_and_grad(
                                lambda cp: iq_loss_fn(cp, policy_state.params, entropy_coefficient_state.params,
                                                      states_b, actions_b, next_states_b, terminations_b,
                                                      expert_states, expert_actions, expert_next_states, expert_absorbing,
                                                      demo_key, vnext_key, v_key, gp_key),
                                has_aux=True
                            )(critic_state.params)

                            critic_state = critic_state.apply_gradients(grads=critic_grads)
                            critic_state = critic_state.replace(
                                target_params=optax.incremental_update(critic_state.params, critic_state.target_params, self.tau)
                            )
                            iq_metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_grads)
                            return (policy_state, critic_state, entropy_coefficient_state, replay_buffer, (expert_states, expert_actions, expert_next_states, expert_absorbing), key), iq_metrics

                        (policy_state, critic_state, entropy_coefficient_state, replay_buffer, (expert_states, expert_actions, expert_next_states, expert_absorbing), key), iq_metrics_seq = jax.lax.scan(
                            iq_update_step,
                            (policy_state, critic_state, entropy_coefficient_state, replay_buffer, (expert_states, expert_actions, expert_next_states, expert_absorbing), key),
                            jnp.arange(self.nr_q_updates_per_step)
                        )
                        iq_metrics = tree.map_structure(lambda x: jnp.mean(x), iq_metrics_seq)


                        """ SAC Policy Update"""
                        def sac_loss_fn(policy_params, critic_params, entropy_coefficient_params, state, next_state, action, key1):
                            # Entropy regularizer
                            alpha_with_grad = self.entropy_coefficient.apply(entropy_coefficient_params)
                            alpha = stop_gradient(alpha_with_grad)

                            # Policy loss
                            dist2 = self.policy.apply(policy_params, state)
                            current_action = dist2.sample(seed=key1)
                            current_log_prob = dist2.log_prob(current_action)
                            entropy = stop_gradient(-current_log_prob)

                            q = self.critic.apply(stop_gradient(critic_params), state, current_action)
                            min_q = jnp.min(q)

                            policy_loss = alpha * current_log_prob - min_q

                            # Entropy loss
                            entropy_loss = alpha_with_grad * (entropy - self.target_entropy)

                            # Combine losses
                            loss = policy_loss + entropy_loss

                            # Create metrics
                            metrics = {
                                "loss/policy_loss": policy_loss,
                                "loss/entropy_loss": entropy_loss,
                                "entropy/entropy": entropy,
                                "entropy/alpha": alpha,
                                "q_value/q_value": min_q,
                            }

                            return loss, (metrics)
                        

                        vmap_sac_loss_fn = jax.vmap(sac_loss_fn, in_axes=(None, None, None, 0, 0, 0, 0), out_axes=0)
                        safe_mean = lambda x: jnp.mean(x) if x is not None else x
                        mean_vmapped_sac_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_sac_loss_fn(*a, **k))
                        grad_sac_loss_fn = jax.value_and_grad(mean_vmapped_sac_loss_fn, argnums=(0, 2), has_aux=True)

                        keys = jax.random.split(key, (self.batch_size) + 2)
                        key, replay_buffer_key, update_keys = keys[0], keys[1], keys[2:]

                        idx1 = jax.random.randint(replay_buffer_key, (self.batch_size,), 0, replay_buffer["size"])
                        idx2 = jax.random.randint(replay_buffer_key, (self.batch_size,), 0, self.nr_envs)
                        states = replay_buffer["states"][idx1, idx2]
                        next_states = replay_buffer["next_states"][idx1, idx2]
                        actions = replay_buffer["actions"][idx1, idx2]

                        (loss, (sac_metrics)), (policy_gradients, entropy_gradients) = grad_sac_loss_fn(
                            policy_state.params, critic_state.params, entropy_coefficient_state.params,
                            states, next_states, actions, update_keys)

                        policy_state = policy_state.apply_gradients(grads=policy_gradients)
                        entropy_coefficient_state = entropy_coefficient_state.apply_gradients(grads=entropy_gradients)
                        sac_metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
                        sac_metrics["gradients/entropy_grad_norm"] = optax.global_norm(entropy_gradients)

                        metrics = {**sac_metrics, **iq_metrics}

                        return (policy_state, critic_state, entropy_coefficient_state, replay_buffer, (expert_states, expert_actions, expert_next_states, expert_absorbing), env_state, key), (env_state.info, metrics)


                    key, subkey = jax.random.split(key)
                    learning_iteration_carry, info_and_optimization_metrics = jax.lax.scan(learning_iteration, (policy_state, critic_state, entropy_coefficient_state, replay_buffer, (expert_states, expert_actions, expert_next_states, expert_absorbing), env_state, subkey), jnp.arange(self.nr_updates_per_logging_iteration))
                    policy_state, critic_state, entropy_coefficient_state, replay_buffer, (expert_states, expert_actions, expert_next_states, expert_absorbing), env_state, key = learning_iteration_carry
                    infos, optimization_metrics = info_and_optimization_metrics
                    infos = {key: jnp.mean(infos[key]) for key in infos}
                    optimization_metrics = {key: jnp.mean(optimization_metrics[key]) for key in optimization_metrics}


                    # Logging
                    combined_metrics = {**infos, **optimization_metrics}
                    combined_metrics = tree.map_structure(lambda x: jnp.mean(x), combined_metrics)

                    def callback(carry):
                        metrics, logging_iteration_step, nr_update_iteration, parallel_seed_id = carry
                        current_time = time.time()
                        metrics["time/sps"] = int((self.nr_envs * self.nr_updates_per_logging_iteration) / (current_time - self.last_time[parallel_seed_id]))
                        self.last_time[parallel_seed_id] = current_time
                        global_step = int(nr_update_iteration.item() * self.nr_envs)
                        metrics["steps/nr_env_steps"] = global_step
                        metrics["steps/nr_updates"] = nr_update_iteration.item()
                        is_last_logging_before_eval = self.evaluation_active and (logging_iteration_step + 1 == self.nr_loggings_per_eval_save_iteration)
                        self.start_logging(global_step)
                        for key, value in metrics.items():
                            self.log(f"{key}", np.asarray(value), global_step)
                        self.end_logging(wandb_commit=not is_last_logging_before_eval)

                    nr_update_iteration = (eval_save_iteration_step * self.nr_loggings_per_eval_save_iteration * self.nr_updates_per_logging_iteration) + (logging_iteration_step+1) * self.nr_updates_per_logging_iteration
                    jax.debug.callback(callback, (combined_metrics, logging_iteration_step, nr_update_iteration, parallel_seed_id))

                    return (policy_state, critic_state, entropy_coefficient_state, replay_buffer, (expert_states, expert_actions, expert_next_states, expert_absorbing), env_state, key), None

                key, subkey = jax.random.split(key)
                logging_iteration_carry, _ = jax.lax.scan(logging_iteration, (policy_state, critic_state, entropy_coefficient_state, replay_buffer, (expert_states, expert_actions, expert_next_states, expert_absorbing), env_state, subkey), jnp.arange(self.nr_loggings_per_eval_save_iteration))
                policy_state, critic_state, entropy_coefficient_state, replay_buffer, (expert_states, expert_actions, expert_next_states, expert_absorbing), env_state, key = logging_iteration_carry


                # Evaluating
                if self.evaluation_active:
                    def single_eval_rollout(carry, _):
                        policy_state, eval_env_state = carry
                        dist_eval = self.policy.apply(policy_state.params, eval_env_state.next_observation)
                        eval_action = jnp.tanh(dist_eval.distribution.loc)
                        eval_processed_action = self.get_processed_action(eval_action)
                        eval_env_state = self.eval_env.step(eval_env_state, eval_processed_action)
                        return (policy_state, eval_env_state), None

                    key, reset_key = jax.random.split(key)
                    reset_keys = jax.random.split(reset_key, self.nr_envs)
                    eval_env_state = self.eval_env.reset(reset_keys, True)
                    (policy_state, eval_env_state), _ = jax.lax.scan(single_eval_rollout, (policy_state, eval_env_state), jnp.arange(self.horizon))

                    eval_metrics = {
                        "eval/episode_return": jnp.mean(eval_env_state.info["rollout/episode_return"]),
                        "eval/episode_length": jnp.mean(eval_env_state.info["rollout/episode_length"]),
                    }

                    def eval_callback(args):
                        metrics, eval_save_iteration_step = args
                        global_step = int((eval_save_iteration_step.item() + 1) * self.evaluation_and_save_frequency)
                        self.start_logging(global_step)
                        for key, value in metrics.items():
                            self.log(f"{key}", np.asarray(value), global_step)
                        self.end_logging()

                    jax.debug.callback(eval_callback, (eval_metrics, eval_save_iteration_step))


                # Saving
                if self.save_model:
                    def save_with_check(policy_state, critic_state, entropy_coefficient_state):
                        self.save(policy_state, critic_state, entropy_coefficient_state)
                    jax.debug.callback(save_with_check, policy_state, critic_state, entropy_coefficient_state)


                return (policy_state, critic_state, entropy_coefficient_state, replay_buffer, (expert_states, expert_actions, expert_next_states, expert_absorbing), env_state, key), None

            jax.lax.scan(eval_save_iteration, (policy_state, critic_state, entropy_coefficient_state, replay_buffer, (expert_states, expert_actions, expert_next_states, expert_absorbing), env_state, key), jnp.arange(self.nr_eval_save_iterations))
            

        self.key, subkey = jax.random.split(self.key)
        seed_keys = jax.random.split(subkey, self.nr_parallel_seeds)
        train_function = jax.jit(jax.vmap(jitable_train_function))
        self.last_time = [time.time() for _ in range(self.nr_parallel_seeds)]
        self.start_time = deepcopy(self.last_time)
        jax.block_until_ready(train_function(seed_keys, jnp.arange(self.nr_parallel_seeds)))
        rlx_logger.info(f"Average time: {max([time.time() - t for t in self.start_time]):.2f} s")


    def log(self, name, value, step):
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_wandb:
            self.wandb_log_cache[name] = value
        if self.track_console:
            self.log_console(name, value)


    def log_console(self, name, value):
        value = np.format_float_positional(value, trim="-")
        rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │", flush=False)


    def start_logging(self, step):
        if self.track_wandb:
            self.wandb_log_cache = {"global_step": int(step)}
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐", flush=False)
        else:
            rlx_logger.info(f"Step: {step}")


    def end_logging(self, wandb_commit=True):
        if self.track_wandb:
            wandb.log(self.wandb_log_cache, commit=wandb_commit)
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")


    def save(self, policy_state, critic_state, entropy_coefficient_state):
        checkpoint = {
            "policy": policy_state,
            "critic": critic_state,
            "entropy_coefficient": entropy_coefficient_state,
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.latest_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
        with open(f"{self.save_path}/tmp/config_algorithm.json", "w") as f:
            json.dump(self.config.algorithm.to_dict(), f)
        shutil.make_archive(f"{self.save_path}/{self.latest_model_file_name}", "zip", f"{self.save_path}/tmp")
        os.rename(f"{self.save_path}/{self.latest_model_file_name}.zip", f"{self.save_path}/{self.latest_model_file_name}")
        shutil.rmtree(f"{self.save_path}/tmp")

        if self.track_wandb:
            wandb.save(f"{self.save_path}/{self.latest_model_file_name}", base_path=self.save_path)


    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = os.path.abspath("/".join(splitted_path[:-1]))
        checkpoint_file_name = splitted_path[-1]
        shutil.unpack_archive(f"{checkpoint_dir}/{checkpoint_file_name}", f"{checkpoint_dir}/tmp", "zip")
        checkpoint_dir = f"{checkpoint_dir}/tmp"

        loaded_algorithm_config = json.load(open(f"{checkpoint_dir}/config_algorithm.json", "r"))
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = IQ_SAC(config, train_env, eval_env, run_path, writer)

        target = {
            "policy": model.policy_state,
            "critic": model.critic_state,
            "entropy_coefficient": model.entropy_coefficient_state,
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)

        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]
        model.entropy_coefficient_state = checkpoint["entropy_coefficient"]

        shutil.rmtree(checkpoint_dir)

        return model
    

    def test(self, episodes):
        rlx_logger.info("Testing runs infinitely. The episodes parameter is ignored.")

        @jax.jit
        def rollout(env_state, key):
            dist = self.policy.apply(self.policy_state.params, env_state.next_observation)
            action = jnp.tanh(dist.distribution.loc)
            processed_action = self.get_processed_action(action)
            env_state = self.eval_env.step(env_state, processed_action)
            return env_state, key

        self.key, subkey = jax.random.split(self.key)
        reset_keys = jax.random.split(subkey, self.nr_envs)
        env_state = self.eval_env.reset(reset_keys, True)
        while True:
            env_state, self.key = rollout(env_state, self.key)
            if self.render:
                env_state = self.eval_env.render(env_state)
    

    def general_properties():
        return GeneralProperties
