import ast
from omegaconf import open_dict
import warnings
from dataclasses import dataclass, replace
from typing import Any
from omegaconf import DictConfig, OmegaConf, ListConfig

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
import flax
import flax.linen as nn
import optax
import wandb
import tree
from jax.lax import stop_gradient

from trust_region_irl.environments.loco_mjx.base_algorithm import JaxRLAlgorithmBase, AgentConfBase, AgentStateBase
from loco_mujoco.algorithms import (FullyConnectedNet, Transition, TrainState,
                                    TrainStateBuffer, MetricHandlerTransition)
from loco_mujoco.core.wrappers import LogWrapper, NStepWrapper, LogEnvState, VecEnv, NormalizeVecReward, SummaryMetrics
from loco_mujoco.utils import MetricsHandler, ValidationSummary
from loco_mujoco.trajectory import TrajectoryTransitions, Trajectory

import os
from trust_region_irl.algorithms.iq_sac.flax_loco_mjx.general_properties import GeneralProperties
from trust_region_irl.algorithms.iq_sac.flax_loco_mjx.policy import Actor, Critic
from trust_region_irl.algorithms.iq_sac.flax_loco_mjx.entropy_coefficient import EntropyCoefficient, ConstantEntropyCoefficient
import logging
from typing import NamedTuple
rlx_logger = logging.getLogger("rl_x")


def flatten_dict(d):
    flat = {}
    for k, v in d.items():
        if hasattr(v, "to_dict"):   # handle ConfigDict
            v = v.to_dict()
        if isinstance(v, dict):
            flat.update(flatten_dict(v))
        else:
            flat[k] = v
    return flat

class RLTrainState(TrainState):
    target_params: flax.core.FrozenDict

@dataclass(frozen=True)
class IQAgentConf(AgentConfBase):
    config: DictConfig
    network: Actor
    critic: Critic
    entropy_coefficient: Any
    tx: Any
    critic_tx: Any
    entropy_coefficient_tx: Any
    expert_dataset: TrajectoryTransitions = None

    def add_expert_dataset(self, expert_dataset):
        return replace(self, expert_dataset=expert_dataset)

    def serialize(self):
        """
        Serialize the agent configuration and network configuration.

        Returns:
            Serialized agent configuration as a dictionary.

        """
        _all = {"config": OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True),
                "network": flax.serialization.to_state_dict(self.network),
                "critic": flax.serialization.to_state_dict(self.critic),
                "entropy_coefficient": flax.serialization.to_state_dict(self.entropy_coefficient),
                "expert_dataset": None} # never save dataset
        return _all

    @classmethod
    def from_dict(cls, d):
        config = OmegaConf.create(d["config"])
        tx, critic_tx, entropy_coefficient_tx = IQ_SAC._get_optimizer(config)
        return cls(config=config,
                   network=flax.serialization.from_state_dict(Actor, d["network"]),
                   critic=flax.serialization.from_state_dict(Critic, d["critic"]),
                   entropy_coefficient=flax.serialization.from_state_dict(EntropyCoefficient, d["entropy_coefficient"]),
                   tx=tx, critic_tx=critic_tx, entropy_coefficient_tx=entropy_coefficient_tx,
                   expert_dataset=flax.serialization.from_state_dict(TrajectoryTransitions, d["expert_dataset"]))


@struct.dataclass
class IQAgentState(AgentStateBase):
    train_state: TrainState
    critic_train_state: RLTrainState
    entropy_coefficient_train_state: TrainState

    def serialize(self):
        serialized_train_state = flax.serialization.to_state_dict(self.train_state)
        serialized_critic = flax.serialization.to_state_dict(self.critic_train_state)
        serialized_entropy_coefficient = flax.serialization.to_state_dict(self.entropy_coefficient_train_state)
        return {"train_state": serialized_train_state,
                "critic": serialized_critic,
                "entropy_coefficient": serialized_entropy_coefficient}

    @classmethod
    def from_dict(cls, d, agent_conf):
        train_state = TrainState(apply_fn=agent_conf.network, tx=agent_conf.tx, **d["train_state"])
        critic_state = RLTrainState(apply_fn=agent_conf.critic, tx=agent_conf.critic_tx, **d["critic"])
        entropy_coefficient_train_state = TrainState(apply_fn=agent_conf.entropy_coefficient, tx=agent_conf.entropy_coefficient_tx, **d["entropy_coefficient"])
        return cls(train_state, critic_state, entropy_coefficient_train_state)


class IQ_SAC(JaxRLAlgorithmBase):

    _agent_conf = IQAgentConf
    _agent_state = IQAgentState


    def __init__(self, config, env, eval_env, run_path, writer, skip_init=False) -> None:

        if not skip_init:
            config, expert_dataset = self.prepare_config_and_expert_dataset(config, env)

            agent_conf = self.init_agent_conf(env, config)
            agent_conf = agent_conf.add_expert_dataset(expert_dataset)
            mh = None

            train_fn = self.build_train_fn(env, writer, agent_conf, mh=mh)
            self.train_fn = jax.jit(train_fn)
            self.key = jax.random.PRNGKey(config.seed)
            self.save_model = config.save_model
            self.save_path = os.path.join(run_path, "models")
            self.cached_agent_conf = agent_conf
            
            rlx_logger.info(f"Using device: {jax.default_backend()}")


    def general_properties():
        return GeneralProperties


    def prepare_config_and_expert_dataset(self, config, env):
        merged = flatten_dict(config)
        if "nr_envs" in merged:
            merged["num_envs"] = merged["nr_envs"]
            del merged["nr_envs"]

        config = OmegaConf.create(merged)

        os.environ['XLA_FLAGS'] = (
            '--xla_gpu_triton_gemm_any=True ')

        # load expert training data
        if config.task == "rl":
            expert_file = np.load(config.data_path)
            expert_dataset = TrajectoryTransitions(
                jnp.array(expert_file["states"]),
                jnp.array(expert_file["next_states"]),
                jnp.array(expert_file["absorbing"]),
                jnp.array(expert_file["absorbing"]),
                jnp.array(expert_file["actions"]),
                jnp.array(expert_file["rewards"]),
                )
        else:
            assert config.reward_type != "state-action", "Mocap data does not contain actions! Please choose a discriminator type that does not use actions"

            expert_dataset_path = config.mocap_data_path + f"{config.agent}_{config.task}.npz"
            if not os.path.exists(expert_dataset_path):
                expert_dataset = env.create_dataset()
                expert_dataset = TrajectoryTransitions(
                expert_dataset.observations,
                expert_dataset.next_observations,
                expert_dataset.absorbings,
                expert_dataset.dones,
                jnp.zeros((expert_dataset.observations.shape[0], env.mdp_info.action_space.shape[0])), # dummy actions
                )
                env.th.traj = replace(env.th.traj, transitions=expert_dataset)

                # save trajectory with expert transitions to speed-up loading next time
                new_traj = Trajectory(info=env.th.traj.info, data=env.th.traj.data,
                                    obs_container=env.obs_container, transitions=expert_dataset)
                new_traj.save(expert_dataset_path)
            else:
                # if it exists, load it
                new_traj = Trajectory.load(expert_dataset_path)
                env.load_trajectory(new_traj)
                expert_dataset = env.create_dataset()

        
        return config, expert_dataset



    def load(config, env, run_path, writer, explicitly_set_algorithm_params):
        model = IQ_SAC(config, env, run_path, writer, skip_init=True)

        path = config.runner.load_model
        # Load the agent state from a file
        if isinstance(path, str):
            path = Path(path)
        if not path.is_file():
            raise ValueError(f'Not a file: {path}')
        if path.suffix != model._saved_agent_suffix:
            raise ValueError(f'Not a {model._saved_agent_suffix} file: {path}')
        with open(path, 'rb') as file:
            data = pickle.load(file)

        keys_subset = ["agent_conf", "agent_state"]
        agent_data = {k: v for k, v in data.items() if k in keys_subset}
        buffers = {k: v for k, v in data.items() if k not in keys_subset}

        new_config, expert_dataset = model.prepare_config_and_expert_dataset(config, env) # config passed on while loading

        agent_conf, agent_state = model.from_dict(agent_data)
        agent_conf = agent_conf.add_expert_dataset(expert_dataset)
        config = agent_conf.config # config saved from previous training
        mh = None

        train_fn = model.build_resume_train_fn(env, writer, agent_conf, agent_state, mh=mh)
        model.train_fn = jax.jit(train_fn)
        model.key = jax.random.PRNGKey(config.seed)
        model.save_model = config.save_model
        model.save_path = os.path.join(run_path, "models")

        # cache objects
        model.cached_agent_conf = agent_conf
        model.cached_agent_state = agent_state
        model.cached_env = env
        model.new_config = new_config

        rlx_logger.info(f"Loaded saved model from {path}")
        rlx_logger.info(f"Using device: {jax.default_backend()}")

        return model



    def test(self, episodes):

        agent_conf = self.cached_agent_conf
        agent_state = self.cached_agent_state
        env = self.cached_env
        new_config = self.new_config # config passed on while loading
        deterministic = True
        n_steps = 1000
        n_envs = 1
        record = False
        # use_mujoco = new_config.use_mujoco
        use_mujoco = True

        if use_mujoco:
            self.play_policy_mujoco(env, agent_conf, agent_state, deterministic=deterministic, n_steps=n_steps, record=record)
        else:
            self.play_policy(env, agent_conf, agent_state, deterministic=deterministic, n_steps=n_steps, n_envs=n_envs, record=record)



    def train(self):
        out = self.train_fn(self.key)

        if self.save_model: 
            os.makedirs(self.save_path)

            agent_state = out["agent_state"]
            path = Path(self.save_path)
            path = path / (self.__class__.__name__ + "_saved")
            path = path.with_suffix(self._saved_agent_suffix)
            
            # serialize agent state
            serialized_state = self.serialize(self.cached_agent_conf, agent_state)

            # save
            with open(path, 'wb') as file:
                pickle.dump(serialized_state, file)

            rlx_logger.info(f"\nSaved agent to: {path}\n")



    @classmethod
    def init_agent_conf(cls, env, config):

        with (open_dict(config)):
            config.num_updates = (
                    config.total_timesteps // config.num_envs)
            
            if config.target_entropy == "auto":
                config.target_entropy = -np.prod(env.info.action_space.shape).item()

        # INIT NETWORK
        hidden_layers = config.hidden_layers \
            if isinstance(config.hidden_layers, (list, ListConfig)) \
            else ast.literal_eval(config.hidden_layers)
        if hasattr(config, "actor_obs_group") and config.actor_obs_group is not None:
            actor_obs_ind = env.obs_container.get_obs_ind_by_group(config.actor_obs_group)
        else:
            actor_obs_ind = jnp.arange(env.mdp_info.observation_space.shape[0])
        if hasattr(config, "critic_obs_group") and config.critic_obs_group is not None:
            critic_obs_ind = env.obs_container.get_obs_ind_by_group(config.critic_obs_group)
        else:
            critic_obs_ind = jnp.arange(env.mdp_info.observation_space.shape[0])
        if hasattr(config, "len_obs_history") and config.len_obs_history > 1:
            obs_len = env.info.observation_space.shape[0]
            actor_obs_ind = jnp.concatenate([actor_obs_ind + i*obs_len
                                             for i in range(config.len_obs_history)])
            critic_obs_ind = jnp.concatenate([critic_obs_ind + i*obs_len
                                              for i in range(config.len_obs_history)])
        network = Actor(
            env.info.action_space.shape[0],
            activation=config.activation,
            init_std=config.init_std,
            learnable_std=config.learnable_std,
            hidden_layer_dims=hidden_layers,
            actor_obs_ind=tuple(actor_obs_ind.tolist()), # must be hashable
        )
        
        critic = Critic(
            nr_critics=2,
            activation=config.activation,
            hidden_layer_dims=hidden_layers,
        )

        if config.learn_ent_coeff:
            entropy_coefficient = EntropyCoefficient(config.init_ent_coeff)
        else:
            entropy_coefficient = ConstantEntropyCoefficient(config.init_ent_coeff)

        # set up optimizers
        tx, critic_tx, entropy_coefficient_tx = cls._get_optimizer(config)

        return cls._agent_conf(config, network, critic, entropy_coefficient, tx, critic_tx, entropy_coefficient_tx)

    @classmethod
    def _get_optimizer(cls, config):

        if config.anneal_lr:
            tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adamw(weight_decay=config.weight_decay, eps=1e-5,
                            learning_rate=lambda count: cls._iq_linear_lr_schedule(count,
                                                                                config.nr_q_updates_per_step, config.lr,
                                                                                config.num_updates))
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adamw(config.lr, weight_decay=config.weight_decay, eps=1e-5),
            )

        critic_tx = optax.chain(
            optax.clip_by_global_norm(config.lr),
            optax.adamw(config.lr, weight_decay=config.weight_decay, eps=1e-5),
        )

        entropy_coefficient_tx = optax.inject_hyperparams(optax.adamw)(learning_rate=config.lr)

        return tx, critic_tx, entropy_coefficient_tx

    @classmethod
    def _train_fn(cls, rng, env, writer,
                  agent_conf: IQAgentConf,
                  agent_state: IQAgentState = None,
                  mh: MetricsHandler = None):

        # extract static agent info
        config, network, critic, entropy_coefficient, tx, critic_tx, entropy_coefficient_tx, expert_dataset =\
            (agent_conf.config, agent_conf.network,
             agent_conf.critic, agent_conf.entropy_coefficient, agent_conf.tx, agent_conf.critic_tx, agent_conf.entropy_coefficient_tx, agent_conf.expert_dataset)

        # extract current agent state
        if agent_state is not None:
            train_state, critic_train_state, entropy_coefficient_train_state = agent_state.train_state, agent_state.critic_train_state, agent_state.entropy_coefficient_train_state
        else:
            train_state, critic_train_state, entropy_coefficient_train_state = None, None, None

        if train_state is None:
            rng, _rng1, _rng2, _rng3 = jax.random.split(rng, 4)
            init_x = jnp.zeros(env.info.observation_space.shape)
            init_a = jnp.zeros(env.info.action_space.shape)
            network_params = network.init(_rng1, init_x)
            critic_params = critic.init(_rng2, init_x, init_a)
            target_params = critic.init(_rng2, init_x, init_a)
            entropy_coefficient_params = entropy_coefficient.init(_rng3)

        else:
            network_params = None
            critic_params = None
            entropy_coefficient_params = None
            # raise NotImplementedError("Loading of train state not implemented yet.")

        # init new train states from old params
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params["params"] if train_state is None else train_state.params,
            run_stats=network_params["run_stats"] if train_state is None else train_state.run_stats,
            tx=tx,
        )

        critic_train_state = RLTrainState.create(
            apply_fn=critic.apply,
            params=critic_params["params"] if critic_train_state is None else critic_train_state.params,
            target_params=target_params["params"] if critic_train_state is None else critic_train_state.target_params,
            run_stats=critic_params["run_stats"] if critic_train_state is None else critic_train_state.run_stats,
            tx=critic_tx,
        )

        entropy_coefficient_train_state = TrainState.create(
            apply_fn=entropy_coefficient.apply,
            params=entropy_coefficient_params["params"] if entropy_coefficient_train_state is None else entropy_coefficient_train_state.params,
            run_stats=None,
            tx=entropy_coefficient_tx,
        )

        env = cls._wrap_env(env, config)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs)
        obsv, env_state = env.reset(reset_rng)

        train_state_buffer = TrainStateBuffer.create(train_state, 10)


        # REPLAY BUFFER
        capacity = int(config.buffer_size // config.num_envs)
        obs_buffer = jnp.zeros((capacity, config.num_envs) + (init_x.shape[0],), dtype=jnp.float32)
        next_obs_buffer = jnp.zeros((capacity, config.num_envs) + (init_x.shape[0],), dtype=jnp.float32)
        actions_buffer = jnp.zeros((capacity, config.num_envs) + (init_a.shape[0],), dtype=jnp.float32)
        rewards_buffer = jnp.zeros((capacity, config.num_envs), dtype=jnp.float32)
        absorbings_buffer = jnp.zeros((capacity, config.num_envs), dtype=jnp.float32)
        replay_buffer = {
            "observations": obs_buffer,
            "next_observations": next_obs_buffer,
            "actions": actions_buffer,
            "rewards": rewards_buffer,
            "absorbings": absorbings_buffer,
            "pos": jnp.zeros((), dtype=jnp.int32),
            "size": jnp.zeros((), dtype=jnp.int32)
        }


        # PRE-FILL REPLAY BUFFER
        rng, _rng = jax.random.split(rng)
        prefill_iterations = int(np.ceil(config.learning_starts // config.num_envs)) if config.learning_starts > 0 else 0
        if prefill_iterations > 0:
            def fill_replay_buffer(carry, _):
                train_state, env_state, replay_buffer, obsv, rng = carry

                observation = obsv

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, _ = network.apply({'params': train_state.params,
                                                  'run_stats': train_state.run_stats},
                                                 obsv, mutable=["run_stats"])
                action = pi.sample(seed=_rng)

                # STEP ENV
                next_obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)

                replay_buffer["observations"] = replay_buffer["observations"].at[replay_buffer["pos"]].set(observation)
                replay_buffer["next_observations"] = replay_buffer["next_observations"].at[replay_buffer["pos"]].set(next_obsv)
                replay_buffer["actions"] = replay_buffer["actions"].at[replay_buffer["pos"]].set(action)
                replay_buffer["absorbings"] = replay_buffer["absorbings"].at[replay_buffer["pos"]].set(absorbing)
                replay_buffer["pos"] = (replay_buffer["pos"] + 1) % capacity
                replay_buffer["size"] = jnp.minimum(replay_buffer["size"] + 1, capacity)

                return (train_state, env_state, replay_buffer, next_obsv, rng), None

            (train_state, env_state, replay_buffer, obsv, _), _ = jax.lax.scan(fill_replay_buffer, (train_state, env_state, replay_buffer, obsv, _rng), jnp.arange(prefill_iterations))


        # TRAIN LOOP
        def _update_step(runner_state, _):
            train_state, critic_train_state, entropy_coefficient_train_state, env_state, last_obs, train_state_buffer, replay_buffer, expert_dataset, rng = runner_state
 
            # Take action and step env
            rng, _rng = jax.random.split(rng)
            pi, updates = network.apply({'params': train_state.params, 'run_stats': train_state.run_stats}, last_obs, mutable=["run_stats"])
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            train_state = train_state.replace(run_stats=updates["run_stats"])
            obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)

            # Add to replay buffer
            replay_buffer["observations"] = replay_buffer["observations"].at[replay_buffer["pos"]].set(last_obs)
            replay_buffer["next_observations"] = replay_buffer["next_observations"].at[replay_buffer["pos"]].set(obsv)
            replay_buffer["actions"] = replay_buffer["actions"].at[replay_buffer["pos"]].set(action)
            replay_buffer["absorbings"] = replay_buffer["absorbings"].at[replay_buffer["pos"]].set(absorbing)
            replay_buffer["pos"] = (replay_buffer["pos"] + 1) % capacity
            replay_buffer["size"] = jnp.minimum(replay_buffer["size"] + 1, capacity)

            def get_v(critic_params, policy_params, ent_params, states, key, policy_run_stats, critic_run_stats):
                alpha_with_grad = entropy_coefficient.apply({"params": ent_params})
                alpha = stop_gradient(alpha_with_grad)

                pi, _ = network.apply({"params": policy_params, "run_stats": policy_run_stats}, states, mutable=["run_stats"])
                actions = pi.sample(seed=key)
                log_prob = pi.log_prob(actions)[..., None]

                q_values, _ = critic.apply({"params": critic_params, "run_stats": critic_run_stats}, states, actions, mutable=["run_stats"])
                q_values = q_values.squeeze(-1).min(axis=0)[:, None]

                v_values = q_values - alpha * log_prob
                return stop_gradient(v_values)

            def regularizer_loss(absorbing, reward, gamma, reg_mult, treat_absorbing_states=False):
                reg_absorbing = absorbing if treat_absorbing_states else jnp.zeros_like(absorbing)
                chi2 = ((1.0 - reg_absorbing) * reg_mult * jnp.square(reward) + reg_absorbing * (1.0 - gamma) * reg_mult * jnp.square(reward)).mean()
                return chi2

            def _iq_update(iq_update_state, _):
                train_state, critic_train_state, entropy_coefficient_train_state, replay_buffer, expert_batch, rng = iq_update_state

                gamma = jnp.asarray(config.gamma, dtype=jnp.float32)
                reg_mult = config.reg_mult
                use_lsiq = getattr(config, "use_lsiq", False)
                v0_loss_flag = getattr(config, "v0_loss", False)
                Q_max = 1.0 / (reg_mult * (1.0 - gamma))
                Q_min = -Q_max
                batch_size = config.batch_size

                rng, _rng1, _rng_keys = jax.random.split(rng, 3)

                idx_time = jax.random.randint(_rng1, (batch_size,), 0, replay_buffer["size"])
                idx_env = jax.random.randint(_rng1, (batch_size,), 0, config.num_envs)

                states = replay_buffer["observations"][idx_time, idx_env]
                next_states = replay_buffer["next_observations"][idx_time, idx_env]
                actions = replay_buffer["actions"][idx_time, idx_env]
                terminations = replay_buffer["absorbings"][idx_time, idx_env]

                # states, next_states, actions, terminations = replay_batch
                expert_states, expert_next_states, expert_actions, expert_terminations = expert_batch

                # https://github.com/Div-Infinity/IQ-Learn/blob/1f5492dd26348ef11dea3467037baf4eff65c178/iq_learn/train_iq.py#L214
                # https://arxiv.org/pdf/2106.12142#page=14
                if config.state_only:
                    expert_actions = actions

                rollout_states, rollout_actions, rollout_next_states = states, actions, next_states
                rollout_absorbing = terminations.astype(jnp.float32)
                num_rollout = rollout_states.shape[0]
                expert_abs = expert_terminations.astype(jnp.float32)

                obs = jnp.concatenate([rollout_states, expert_states], axis=0)
                act = jnp.concatenate([rollout_actions, expert_actions], axis=0)
                next_obs = jnp.concatenate([rollout_next_states, expert_next_states], axis=0)
                absorbing = jnp.concatenate([rollout_absorbing, expert_abs], axis=0)[:, None]
                
                is_expert = jnp.concatenate([jnp.zeros((num_rollout,), dtype=bool), jnp.ones((num_rollout,), dtype=bool)], axis=0)
                is_expert_f = is_expert.astype(jnp.float32)
                expert_denom = is_expert_f.sum() + 1e-8

                key_v_next, key_v = jax.random.split(_rng_keys)

                def loss_fn(critic_params):
                    q_values, updates = critic.apply(
                        {"params": critic_params, "run_stats": critic_train_state.run_stats},
                        obs,
                        act,
                        mutable=["run_stats"],
                    )
                    q_values = q_values.squeeze(-1).min(axis=0)[:, None]

                    target_params = jax.lax.cond(
                        config.use_target_q,
                        lambda _: critic_train_state.target_params,
                        lambda _: critic_params,
                        operand=None
                    )
                    next_v = get_v(target_params, train_state.params, entropy_coefficient_train_state.params, next_obs, key_v_next, train_state.run_stats, critic_train_state.run_stats)
                    
                    if use_lsiq:
                        y = (1.0 - absorbing) * gamma * jnp.clip(next_v, Q_min, Q_max)
                    else:
                        y = (1.0 - absorbing) * gamma * next_v

                    reward = q_values - y
                    reward_flat = reward.squeeze(-1)
                    exp_reward_mean = (reward_flat * is_expert_f).sum() / expert_denom

                    if use_lsiq:
                        loss_term1 = optax.losses.squared_error(q_values * is_expert_f[:, None], jnp.ones_like(q_values) * Q_max).mean()
                    else:
                        loss_term1 = -exp_reward_mean

                    V = get_v(critic_params, train_state.params, entropy_coefficient_train_state.params, obs, key_v, train_state.run_stats, critic_train_state.run_stats)
                    value = (V - y).squeeze(-1)

                    if v0_loss_flag:
                        V_flat = V.squeeze(-1)
                        V_exp_mean = (V_flat * is_expert_f).sum() / expert_denom
                        loss_term2 = (1.0 - gamma) * V_exp_mean
                    else:
                        loss_term2 = value.mean()

                    chi2 = regularizer_loss(absorbing, reward, gamma, reg_mult, treat_absorbing_states=True)
                    loss_Q = loss_term1 + loss_term2 + chi2

                    diff_exp = reward_flat - exp_reward_mean
                    exp_reward_var = ((diff_exp ** 2) * is_expert_f).sum() / expert_denom
                    exp_reward_std = jnp.sqrt(exp_reward_var)

                    metrics = {
                        "iq/loss_q": loss_Q,
                        "iq/loss_term1_expert": loss_term1,
                        "iq/loss_term2_value": loss_term2,
                        "iq/chi2_loss": chi2,
                        "iq/reward_mean": reward_flat.mean(),
                        "iq/reward_expert_mean": exp_reward_mean,
                        "iq/reward_expert_std": exp_reward_std,
                    }
                    return loss_Q, (updates, metrics)

                (loss_Q, (updates, metrics)), critic_grads = jax.value_and_grad(loss_fn, has_aux=True)(critic_train_state.params)
                new_critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
                new_target_params = optax.incremental_update(
                    new_critic_train_state.params, new_critic_train_state.target_params, config.tau
                )
                new_critic_train_state = new_critic_train_state.replace(target_params=new_target_params)
                new_critic_train_state = new_critic_train_state.replace(run_stats=updates["run_stats"])

                metrics = {k: jnp.mean(v) for k, v in metrics.items()}
                metrics["gradients/q_grad_norm"] = optax.global_norm(critic_grads)

                new_state = (train_state, new_critic_train_state, entropy_coefficient_train_state, replay_buffer, expert_batch, rng)
                return new_state, metrics

            def _sac_update(sac_update_state, _):
                train_state, critic_train_state, entropy_coefficient_train_state, replay_buffer, rng = sac_update_state

                batch_size = config.batch_size
                target_entropy = float(config.target_entropy)

                rng, _rng1, _rng_keys = jax.random.split(rng, 3)
                idx_time = jax.random.randint(_rng1, (batch_size,), 0, replay_buffer["size"])
                idx_env = jax.random.randint(_rng1, (batch_size,), 0, config.num_envs)

                states = replay_buffer["observations"][idx_time, idx_env]
                next_states = replay_buffer["next_observations"][idx_time, idx_env]
                actions = replay_buffer["actions"][idx_time, idx_env]

                def loss_fn(policy_params, critic_params, entropy_params, state, next_state, action, key1):
                    alpha_with_grad = entropy_coefficient.apply({"params": entropy_params})
                    alpha = stop_gradient(alpha_with_grad)

                    pi, _ = network.apply({"params": policy_params, "run_stats": train_state.run_stats}, state, mutable=["run_stats"])
                    current_action = pi.sample(seed=key1)
                    current_log_prob = pi.log_prob(current_action)
                    entropy_val = stop_gradient(-current_log_prob)

                    q_values, _ = critic.apply({"params": stop_gradient(critic_params), "run_stats": critic_train_state.run_stats}, state[None, ...], current_action[None, ...], mutable=["run_stats"])
                    min_q = q_values.squeeze(-1).min(axis=0)

                    policy_loss = alpha * current_log_prob - min_q
                    entropy_loss = alpha_with_grad * (entropy_val - target_entropy)
                    loss = policy_loss + entropy_loss

                    metrics = {"sac/policy_loss": policy_loss, "sac/entropy_loss": entropy_loss, "sac/entropy": entropy_val, "sac/alpha": alpha, "sac/min_q": min_q}
                    return loss, metrics

                vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, None, 0, 0, 0, 0), out_axes=0)
                safe_mean = lambda x: jnp.mean(x) if x is not None else x

                def mean_vmapped_loss_fn(policy_params, critic_params, entropy_params, states, next_states, actions, keys):
                    loss, metrics = vmap_loss_fn(policy_params, critic_params, entropy_params, states, next_states, actions, keys)
                    mean_loss = loss.mean()
                    mean_metrics = tree.map_structure(safe_mean, metrics)
                    return mean_loss, mean_metrics

                grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 2), has_aux=True)

                keys = jax.random.split(_rng_keys, batch_size + 1)
                rng = keys[0]
                keys1 = keys[1:]

                (loss, metrics), (policy_grads, entropy_grads) = grad_loss_fn(
                    train_state.params,
                    critic_train_state.params,
                    entropy_coefficient_train_state.params,
                    states,
                    next_states,
                    actions,
                    keys1,
                )

                new_train_state = train_state.apply_gradients(grads=policy_grads)
                new_entropy_state = entropy_coefficient_train_state.apply_gradients(grads=entropy_grads)

                metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_grads)
                metrics["gradients/entropy_grad_norm"] = optax.global_norm(entropy_grads)

                new_state = (new_train_state, critic_train_state, new_entropy_state, replay_buffer, rng)
                return new_state, metrics


            def _get_one_batch(data, batch_size, rng):
                d0, d1, d2, d3 = data
                idx = jax.random.randint(rng, shape=(batch_size,), minval=0, maxval=d0.shape[0])
                return d0[idx], d1[idx], d2[idx], d3[idx]

            rng, _rng1, _rng2 = jax.random.split(rng, 3)
            expert_batch = _get_one_batch(
                (expert_dataset.observations,
                expert_dataset.next_observations,
                expert_dataset.actions,
                expert_dataset.absorbings),
                config.batch_size,
                _rng1,
            )

            iq_update_state = (train_state, critic_train_state, entropy_coefficient_train_state, replay_buffer, expert_batch, rng)
            iq_update_state, iq_metrics_seq = jax.lax.scan(_iq_update, iq_update_state, None, config.nr_q_updates_per_step)
            train_state, critic_train_state, entropy_coefficient_train_state, replay_buffer, expert_batch, rng = iq_update_state
            iq_metrics = tree.map_structure(lambda x: x.mean(axis=0), iq_metrics_seq)

            sac_update_state = (train_state, critic_train_state, entropy_coefficient_train_state, replay_buffer, rng)
            sac_update_state, sac_metrics_seq = jax.lax.scan(_sac_update, sac_update_state, None, 1)
            train_state, critic_train_state, entropy_coefficient_train_state, replay_buffer, rng = sac_update_state
            sac_metrics = tree.map_structure(lambda x: x.mean(axis=0), sac_metrics_seq)

            def metrics_callback(args):
                log_step, env_metrics, iq_metrics, sac_metrics, log_std = args
                step = int(jax.device_get(log_step))

                done = np.array(env_metrics.done)
                rets = np.array(env_metrics.returned_episode_returns)
                lens = np.array(env_metrics.returned_episode_lengths)
                timesteps = np.array(env_metrics.timestep)

                if done.sum() > 0:
                    mean_ret = float((rets * done).sum() / done.sum())
                    mean_len = float((lens * done).sum() / done.sum())
                    max_timestep = int((timesteps * done).max() * config.num_envs)
                else:
                    mean_ret = 0.0
                    mean_len = 0.0
                    max_timestep = step

                log_dict = {
                    "rollout/episode_return": mean_ret,
                    "rollout/episode_length": mean_len,
                    "rollout/nr_env_steps": max_timestep,
                    "policy/std": float(np.exp(np.array(log_std)).mean()),
                }

                metrics_all = {}
                metrics_all.update({k: float(np.array(v)) for k, v in iq_metrics.items()})
                metrics_all.update({k: float(np.array(v)) for k, v in sac_metrics.items()})
                log_dict.update(metrics_all)

                rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐", flush=False)
                for name, value in log_dict.items():
                    writer.add_scalar(name, value, step)
                    if wandb.run is not None:
                        wandb.log({name: value, "global_step": int(step)}, commit=False)
                    rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │", flush=False)
                rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")

            log_env_state = env_state.find(LogEnvState)
            logged_metrics = log_env_state.metrics
            log_step = (train_state.step + 1) * config.num_envs

            jax.lax.cond(
                (log_step) % (10 * int(config.num_envs)) == 0,
                lambda _: jax.debug.callback(
                    metrics_callback,
                    (log_step, logged_metrics, iq_metrics, sac_metrics, train_state.params["log_std"]),
                    ),
                lambda _: None,
                operand=None,
            )

            runner_state = (train_state, critic_train_state, entropy_coefficient_train_state, env_state, obsv, train_state_buffer, replay_buffer, expert_dataset, rng)
            return runner_state, (logged_metrics, None)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, critic_train_state, entropy_coefficient_train_state, env_state, obsv, train_state_buffer, replay_buffer, expert_dataset, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config.num_updates
        )

        agent_state = cls._agent_state(train_state=runner_state[0], critic_train_state=runner_state[1], entropy_coefficient_train_state=runner_state[2])

        return {"agent_state": agent_state}


    @classmethod
    def play_policy(cls, env,
                    agent_conf: IQAgentConf,
                    agent_state: IQAgentState,
                    n_envs: int, n_steps=None, render=True,
                    record=False, rng=None, deterministic=False,
                    use_mujoco=False, wrap_env=True,
                    train_state_seed=None):

        if use_mujoco and wrap_env:
            if hasattr(agent_conf.experiment, "len_obs_history"):
                assert agent_conf.experiment.len_obs_history == 1, "len_obs_history must be 1 for mujoco envs."
        if use_mujoco:
            assert n_envs == 1, "Only one mujoco env can be run at a time."

        def sample_actions(ts, obs, _rng):
            y, updates = agent_conf.network.apply({'params': ts.params,
                                                   'run_stats': ts.run_stats},
                                                  obs, mutable=["run_stats"])
            ts = ts.replace(run_stats=updates['run_stats'])  # update stats
            pi, _ = y
            a = pi.sample(seed=_rng)
            return a, ts

        config = agent_conf.config
        train_state = agent_state.train_state

        if deterministic:
            train_state.params["log_std"] = np.ones_like(train_state.params["log_std"]) * -np.inf

        if config.n_seeds > 1:
            assert train_state_seed is not None, ("Loaded train state has multiple seeds. Please specify "
                                                  "train_state_seed for replay.")

            # take the seed queried for evaluation
            train_state = jax.tree.map(lambda x: x[train_state_seed], train_state)

        if not render and n_steps is None and not record:
            warnings.warn("No rendering, no record, no n_steps specified. This will run forever with no effect.")

        # create env
        if wrap_env and not use_mujoco:
            env = cls._wrap_env(env, config)

        if rng is None:
            rng = jax.random.key(0)

        keys = jax.random.split(rng, n_envs + 1)
        rng, env_keys = keys[0], keys[1:]

        plcy_call = jax.jit(sample_actions)

        # reset env
        if use_mujoco:
            obs = env.reset()
            env_state = None
        else:
            obs, env_state = env.reset(env_keys)

        if n_steps is None:
            n_steps = np.iinfo(np.int32).max

        for i in range(n_steps):

            # SAMPLE ACTION
            rng, _rng = jax.random.split(rng)
            action, train_state = plcy_call(train_state, obs, _rng)
            action = jnp.atleast_2d(action)

            # STEP ENV
            if use_mujoco:
                obs, reward, absorbing, done, info = env.step(action)
            else:
                obs, reward, absorbing, done, info, env_state = env.step(env_state, action)

            # RENDER
            if use_mujoco:
                env.render(record=record)
            else:
                env.mjx_render(env_state, record=record)

            # RESET MUJOCO ENV (MJX resets by itself)
            if use_mujoco:
                if done:
                    obs = env.reset()

        env.stop()

    @classmethod
    def play_policy_mujoco(cls, env,
                           agent_conf: IQAgentConf,
                           agent_state: IQAgentState,
                           n_steps=None, render=True,
                           record=False, rng=None, deterministic=False,
                           train_state_seed=None):

        cls.play_policy(env, agent_conf, agent_state, 1, n_steps, render, record, rng, deterministic,
                        True, False, train_state_seed)

    @staticmethod
    def _wrap_env(env, config):

        if "len_obs_history" in config and config.len_obs_history > 1:
            env = NStepWrapper(env, config.len_obs_history)
        env = LogWrapper(env)
        env = VecEnv(env)
        if config.normalize_env:
            env = NormalizeVecReward(env, config.gamma)
        return env

    @classmethod
    def _iq_linear_lr_schedule(cls, count, nr_q_updates_per_step, lr, num_updates):
        frac = (
                1.0
                - (count // (nr_q_updates_per_step))
                / num_updates
        )
        return lr * frac
