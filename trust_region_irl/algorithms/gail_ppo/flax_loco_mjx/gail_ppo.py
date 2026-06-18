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

from trust_region_irl.environments.loco_mjx.base_algorithm import JaxRLAlgorithmBase, AgentConfBase, AgentStateBase
from loco_mujoco.algorithms import (ActorCritic, FullyConnectedNet, Transition, TrainState,
                                    TrainStateBuffer, MetricHandlerTransition)
from loco_mujoco.core.wrappers import LogWrapper, NStepWrapper, LogEnvState, VecEnv, NormalizeVecReward, SummaryMetrics
from loco_mujoco.utils import MetricsHandler, ValidationSummary
from loco_mujoco.trajectory import TrajectoryTransitions, Trajectory

import os
from trust_region_irl.algorithms.gail_ppo.flax_loco_mjx.discriminator import get_discriminator
from trust_region_irl.algorithms.gail_ppo.flax_loco_mjx.general_properties import GeneralProperties
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


class GAILTransition(NamedTuple):
    done: jax.Array
    absorbing: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    obs_n: jax.Array
    info: jax.Array
    traj_state: "TrajState"
    metrics: "Metrics"


@struct.dataclass
class GailSummaryMetrics(SummaryMetrics):
    discriminator_output_policy: float = 0.0
    discriminator_output_expert: float = 0.0
    discriminator_gp_loss: float = 0.0


@dataclass(frozen=True)
class GAILAgentConf(AgentConfBase):
    config: DictConfig
    network: ActorCritic
    discriminator: Any
    tx: Any
    disc_tx: Any
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
                "discriminator": flax.serialization.to_state_dict(self.discriminator),
                "expert_dataset": None} # never save dataset
        return _all

    @classmethod
    def from_dict(cls, d):
        config = OmegaConf.create(d["config"])
        tx, disc_tx = GAIL_PPO._get_optimizer(config)
        return cls(config=config,
                   network=flax.serialization.from_state_dict(ActorCritic, d["network"]),
                   discriminator=flax.serialization.from_state_dict(get_discriminator(config.reward_type), d["discriminator"]),
                   tx=tx, disc_tx=disc_tx,
                   expert_dataset=flax.serialization.from_state_dict(TrajectoryTransitions, d["expert_dataset"]))


@struct.dataclass
class GAILAgentState(AgentStateBase):
    train_state: TrainState
    disc_train_state: TrainState

    def serialize(self):
        serialized_train_state = flax.serialization.to_state_dict(self.train_state)
        serialized_discriminator = flax.serialization.to_state_dict(self.disc_train_state)
        return {"train_state": serialized_train_state,
                "discriminator": serialized_discriminator}

    @classmethod
    def from_dict(cls, d, agent_conf):
        train_state = TrainState(apply_fn=agent_conf.network, tx=agent_conf.tx, **d["train_state"])
        disc_state = TrainState(apply_fn=agent_conf.discriminator, tx=agent_conf.disc_tx, **d["discriminator"])
        return cls(train_state, disc_state)


class GAIL_PPO(JaxRLAlgorithmBase):

    _agent_conf = GAILAgentConf
    _agent_state = GAILAgentState


    def __init__(self, config, env, eval_env, run_path, writer, skip_init=False) -> None:

       if not skip_init:
            config, expert_dataset = self.prepare_config_and_expert_dataset(config, env)

            agent_conf = self.init_agent_conf(env, config)
            agent_conf = agent_conf.add_expert_dataset(expert_dataset)
            mh = MetricsHandler(config, env) if config.validation_active else None

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
        model = GAIL_PPO(config, env, run_path, writer, skip_init=True)

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
        mh = MetricsHandler(config, env) if config.validation_active else None

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
                    config.total_timesteps // config.num_steps // config.num_envs)
            config.minibatch_size = (
                    config.num_envs * config.num_steps // config.num_minibatches)
            config.validation_interval = config.num_updates // config.validation_num
            config.validation_num = int(
                config.num_updates // config.validation_interval)

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
        network = ActorCritic(
            env.info.action_space.shape[0],
            activation=config.activation,
            init_std=config.init_std,
            learnable_std=config.learnable_std,
            hidden_layer_dims=hidden_layers,
            actor_obs_ind=tuple(actor_obs_ind.tolist()), # must be hashable
            critic_obs_ind=tuple(critic_obs_ind.tolist()), # must be hashable
        )
        discriminator = get_discriminator(config.reward_type)(activation=config.activation,
                                          hidden_layer_dims=config.hidden_layers,
                                          output_dim=1, output_activation=None,
                                          use_running_mean_stand=True,
                                          squeeze_output=True,
                                          gamma=config.gamma)

        # set up optimizers
        tx, disc_tx = cls._get_optimizer(config)

        return cls._agent_conf(config, network, discriminator, tx, disc_tx)

    @classmethod
    def _get_optimizer(cls, config):

        if config.anneal_lr:
            tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adamw(weight_decay=config.weight_decay, eps=1e-5,
                            learning_rate=lambda count: cls._linear_lr_schedule(count, config.num_minibatches,
                                                                                config.update_epochs, config.lr,
                                                                                config.num_updates))
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adamw(config.lr, weight_decay=config.weight_decay, eps=1e-5),
            )

        disc_tx = optax.chain(
            optax.clip_by_global_norm(config.lr),
            optax.adamw(config.disc_lr, weight_decay=config.weight_decay, eps=1e-5),
        )

        return tx, disc_tx

    @classmethod
    def _train_fn(cls, rng, env, writer,
                  agent_conf: GAILAgentConf,
                  agent_state: GAILAgentState = None,
                  mh: MetricsHandler = None):

        # extract static agent info
        config, network, discriminator, tx, disc_tx, expert_dataset =\
            (agent_conf.config, agent_conf.network,
             agent_conf.discriminator, agent_conf.tx, agent_conf.disc_tx, agent_conf.expert_dataset)

        # extract current agent state
        if agent_state is not None:
            train_state, disc_train_state = agent_state.train_state, agent_state.disc_train_state
        else:
            train_state, disc_train_state = None, None

        if train_state is None:

            rng, _rng1, _rng2 = jax.random.split(rng, 3)
            init_x = jnp.zeros(env.info.observation_space.shape)
            init_xn = jnp.zeros(env.info.observation_space.shape)
            init_a = jnp.zeros(env.info.action_space.shape)
            init_abs = jnp.array([0.0])
            init_log_prob = jnp.zeros(env.info.action_space.shape)
            network_params = network.init(_rng1, init_x)
            discrim_params = discriminator.init(_rng2, init_x, init_a, init_xn, init_abs, init_log_prob)

        else:
            network_params = None
            discrim_params = None

        # init new train states from old params
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params["params"] if train_state is None else train_state.params,
            run_stats=network_params["run_stats"] if train_state is None else train_state.run_stats,
            tx=tx,
        )

        disc_train_state = TrainState.create(
            apply_fn=discriminator.apply,
            params=discrim_params["params"] if disc_train_state is None else disc_train_state.params,
            run_stats=discrim_params["run_stats"] if disc_train_state is None else disc_train_state.run_stats,
            tx=disc_tx,
        )

        env = cls._wrap_env(env, config)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs)
        obsv, env_state = env.reset(reset_rng)

        train_state_buffer = TrainStateBuffer.create(train_state, config.validation_num)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, disc_train_state, env_state, last_obs, train_state_buffer, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                y, updates = network.apply({'params': train_state.params,
                                                  'run_stats': train_state.run_stats},
                                                 last_obs, mutable=["run_stats"])
                pi, value = y
                train_state = train_state.replace(run_stats=updates['run_stats'])   # update stats
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)

                # GET METRICS
                log_env_state = env_state.find(LogEnvState)
                logged_metrics = log_env_state.metrics

                transition = GAILTransition(
                    done, absorbing, action, value, reward, log_prob, last_obs, obsv, info, env_state.additional_carry.traj_state,
                    logged_metrics
                )
                runner_state = (train_state, disc_train_state, env_state, obsv, train_state_buffer, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # CALCULATE ADVANTAGE
            train_state, disc_train_state, env_state, last_obs, train_state_buffer, rng = runner_state
            y, _ = network.apply({'params': train_state.params,
                                  'run_stats': train_state.run_stats},
                                 last_obs, mutable=["run_stats"])
            pi, last_val = y

            def _calculate_gae(traj_batch, last_val, disc_train_state):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, absorbing, value, reward, obs, act, obs_n, log_prob = (
                        transition.done,
                        transition.absorbing,
                        transition.value,
                        transition.reward,
                        transition.obs,
                        transition.action,
                        transition.obs_n,
                        transition.log_prob,
                    )

                    # predict reward with discriminator
                    discrim_reward = cls._predict_rewards((obs, act, obs_n, absorbing, log_prob), discriminator, disc_train_state)

                    # compute proportion of each reward
                    reward = (config.proportion_env_reward * reward +
                              (1 - config.proportion_env_reward) * discrim_reward)


                    if config.handle_absorbing_states:
                        as_bounds = (env.mdp_info.action_space.high - env.mdp_info.action_space.low).sum()
                        next_log_prob = (1/as_bounds) * jnp.ones_like(log_prob)
                        rewards_next_state = cls._predict_rewards((obs_n, 0.0*act, obs_n, jnp.ones_like(absorbing), next_log_prob), discriminator, disc_train_state)
                        H_terminal = jnp.sum(jnp.log(as_bounds))
                        terminal_tail = (config.gamma / (1.0 - config.gamma)) * (rewards_next_state + config.ent_coef * H_terminal)
                        delta = reward + config.gamma * next_value * (1.0 - absorbing) + (absorbing * terminal_tail) - value
                    else:
                        delta = reward + config.gamma * next_value * (1.0 - absorbing) - value

                    gae = (
                        delta
                        + config.gamma * config.gae_lambda * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, disc_train_state)


            # UPDATE ACTOR & CRITIC NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        y, _ = network.apply({'params': params, 'run_stats': train_state.run_stats},
                                             traj_batch.obs, mutable=["run_stats"])
                        pi, value = y
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config.clip_eps, config.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE PPO ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config.clip_eps,
                                    1.0 + config.clip_eps,
                                )
                                * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config.vf_coef * value_loss
                            - config.ent_coef * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, disc_train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config.minibatch_size * config.num_minibatches
                assert (
                    batch_size == config.num_steps * config.num_envs
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config.num_minibatches, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, disc_train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, disc_train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )
            train_state = update_state[0]
            rng = update_state[-1]

            def _update_discriminator(runner_state, unused):
                disc_train_state, traj_batch, train_state, rng = runner_state
                def _get_one_batch(data, batch_size, rng):
                    data0, data1, data2, data3, data4 = data
                    idx = jax.random.randint(rng, shape=(batch_size,), minval=0, maxval=data0.shape[0])
                    return data0[idx], data1[idx], data2[idx], data3[idx], data4[idx]

                def _discrim_loss(params, disc_train_state, inputs, targets, rng):
                    logits, updates = discriminator.apply({'params': params,
                                                           'run_stats': disc_train_state.run_stats},
                                                          *inputs, mutable=["run_stats"])

                    # update running statistics
                    disc_train_state = disc_train_state.replace(run_stats=updates["run_stats"])

                    # calculate loss
                    total_loss, discrim_out = cls._discriminator_loss(config, logits, targets)

                    # gradient penalty
                    gp_loss = cls._discriminator_gp(config, discriminator, params, disc_train_state, inputs, rng)
                    total_loss += config.gp_lambda * gp_loss

                    # calculate some stats
                    plcy_idxs = jnp.arange(0, config.disc_minibatch_size)
                    exp_idxs = jnp.arange(config.disc_minibatch_size, 2 * config.disc_minibatch_size)
                    discrim_probs_policy = discrim_out[plcy_idxs]
                    discrim_probs_exp = discrim_out[exp_idxs]

                    if config.debug:

                        def callback(discrim_probs_policy, discrim_probs_exp):
                            print(f"Policy Discriminator Output: {jnp.mean(discrim_probs_policy)}")
                            print(f"Expert Discriminator Output: {jnp.mean(discrim_probs_exp)}")

                        jax.debug.callback(callback, discrim_probs_policy, discrim_probs_exp)

                    return total_loss, (disc_train_state, discrim_probs_policy, discrim_probs_exp, gp_loss)

                # Get one batch of policy and expert demonstrations
                rng, _rng1, _rng2, _rng3, _rng4 = jax.random.split(rng, 5)
                batch_size = config.disc_minibatch_size

                obs = traj_batch.obs.reshape((-1, traj_batch.obs.shape[-1]))
                obs_n = traj_batch.obs_n.reshape((-1, traj_batch.obs_n.shape[-1]))
                act = traj_batch.action.reshape((-1, traj_batch.action.shape[-1]))
                absorbing = traj_batch.absorbing.flatten()
                log_prob = traj_batch.log_prob.flatten()

                # log probability of the expert actions for the agent policy on expert states 
                y, _ = network.apply({'params': train_state.params,
                                                  'run_stats': train_state.run_stats},
                                                 expert_dataset.observations, mutable=["run_stats"])
                pi_on_exp, _ = y
                expert_log_prob = pi_on_exp.log_prob(expert_dataset.actions)

                obs_batch, obs_n_batch, act_batch, absorbing_batch, log_prob_batch = _get_one_batch((obs, obs_n, act, absorbing, log_prob), batch_size, _rng1)
                demo_obs_batch, demo_obs_n_batch, demo_act_batch, demo_absorbing_batch, demo_log_prob_batch = _get_one_batch((expert_dataset.observations, 
                                                                                                        expert_dataset.next_observations, 
                                                                                                        expert_dataset.actions, 
                                                                                                        expert_dataset.absorbings,
                                                                                                        expert_log_prob,
                                                                                                        ), 
                                                                                                        batch_size, _rng2)
                
                # Create labels
                plcy_target = jnp.zeros(shape=(obs_batch.shape[0],))
                demo_target = jnp.ones(shape=(demo_obs_batch.shape[0],))

                # concatenate inputs and targets
                inputs = (jnp.concatenate([obs_batch, demo_obs_batch], axis=0),
                        jnp.concatenate([act_batch, demo_act_batch], axis=0),
                        jnp.concatenate([obs_n_batch, demo_obs_n_batch], axis=0),
                        jnp.concatenate([absorbing_batch, demo_absorbing_batch], axis=0),
                        jnp.concatenate([log_prob_batch, demo_log_prob_batch], axis=0)
                        )

                targets = jnp.concatenate([plcy_target, demo_target], axis=0)

                # update discriminator
                grad_fn = jax.value_and_grad(_discrim_loss, has_aux=True)
                (total_loss, (disc_train_state, discrim_probs_policy, discrim_probs_exp, discrim_gp_loss)), grads =\
                    grad_fn(disc_train_state.params, disc_train_state, inputs, targets, _rng3)

                # apply discriminator gradients
                disc_train_state = disc_train_state.apply_gradients(grads=grads)

                runner_state = (disc_train_state, traj_batch, train_state, rng)
                return runner_state, (total_loss, discrim_probs_policy, discrim_probs_exp, discrim_gp_loss)

            (disc_train_state, traj_batch, train_state, rng), (discrim_loss_info, discrim_probs_policy, discrim_probs_exp, discrim_gp_loss) = jax.lax.scan(
                _update_discriminator, (disc_train_state, traj_batch, train_state, rng), xs=None, length=config.n_disc_epochs
            )

            counter = ((train_state.step + 1) // config.num_minibatches) // config.update_epochs

            logged_metrics = traj_batch.metrics
            metric = GailSummaryMetrics(
                discriminator_output_policy=jnp.mean(discrim_probs_policy),
                discriminator_output_expert=jnp.mean(discrim_probs_exp),
                discriminator_gp_loss=jnp.mean(discrim_gp_loss),
                mean_episode_return=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 0.0)) / jnp.sum(logged_metrics.done),
                mean_episode_length=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 0.0)) / jnp.sum(logged_metrics.done),
                max_timestep=jnp.max(logged_metrics.timestep * config.num_envs),
            )

            def _evaluation_step():

                def _eval_env(runner_state, unused):
                    train_state, disc_train_state, env_state, last_obs, train_state_buffer, rng = runner_state

                    # SELECT ACTION
                    rng, _rng = jax.random.split(rng)
                    y, updates = train_state.apply_fn({'params': train_state.params,
                                                       'run_stats': train_state.run_stats},
                                                      last_obs, mutable=["run_stats"])
                    pi, value = y
                    train_state = train_state.replace(run_stats=updates['run_stats'])  # update stats
                    action = pi.sample(seed=_rng)

                    # STEP ENV
                    obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)

                    # GET METRICS
                    log_env_state = env_state.find(LogEnvState)
                    logged_metrics = log_env_state.metrics

                    transition = MetricHandlerTransition(env_state, logged_metrics)

                    runner_state = (train_state, disc_train_state, env_state, obsv, train_state_buffer, rng)
                    return runner_state, transition

                rng = runner_state[-1]
                reset_rng = jax.random.split(rng, config.validation_num_envs)
                obsv, env_state = env.reset(reset_rng)
                runner_state_eval = (train_state, disc_train_state, env_state, obsv, train_state_buffer, rng)

                # do evaluation runs
                _, traj_batch_eval = jax.lax.scan(
                    _eval_env, runner_state_eval, None, config.validation_num_steps
                )

                env_states = traj_batch_eval.env_state

                validation_metrics = mh(env_states)

                return validation_metrics

            if mh is None:
                validation_metrics = ValidationSummary()
            else:
                validation_metrics = jax.lax.cond(counter % config.validation_interval == 0, _evaluation_step,
                                                  mh.get_zero_container)


            # LOGGING
            def metrics_callback(metric):
                rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐", flush=False)

                metric, logged_metrics, loss_info, log_std, discrim_loss = metric
                total_loss, (value_loss, loss_actor, entropy) = loss_info
                log_step = metric.max_timestep.astype('int32')
                log_dict = {
                    "rollout/episode_return": metric.mean_episode_return,
                    "rollout/episode_length": metric.mean_episode_length,
                    "rollout/dones": jnp.mean(logged_metrics.done),
                    "loss/total_loss": jnp.mean(total_loss),
                    "loss/policy_loss": jnp.mean(loss_actor),
                    "loss/critic_loss": jnp.mean(value_loss),
                    "loss/discriminator_gp": metric.discriminator_gp_loss,
                    "loss/discriminator_loss": jnp.mean(discrim_loss),
                    "policy/entropy": jnp.mean(entropy),
                    "policy/std": jnp.mean(jnp.exp(log_std)),
                    "discriminator/policy_pred": metric.discriminator_output_policy,
                    "discriminator/expert_pred": metric.discriminator_output_expert,
                    "rollout/nr_env_steps": log_step,     
                    } 
                for name, value in log_dict.items():
                    if isinstance(value, jax.Array):
                        value = value.__array__()
                    writer.add_scalar(name, value.__array__(), log_step.__array__())
                    if wandb.run is not None:
                        wandb.log({name: value, "global_step": int(log_step)}, commit=False)
                    rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │", flush=False)

                rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")

            jax.debug.callback(metrics_callback, (metric, logged_metrics, loss_info, train_state.params["log_std"], discrim_loss_info))


            if config.debug:
                def callback(metrics):
                    return_values = metrics.returned_episode_returns[metrics.done]
                    timesteps = metrics.timestep[metrics.done] * config.num_envs

                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")

                jax.debug.callback(callback, env_state.metrics)

            # add train state to buffer if needed
            train_state_buffer = jax.lax.cond(counter % config.validation_interval == 0,
                                              lambda x, y: TrainStateBuffer.add(x, y),
                                              lambda x, y: x, train_state_buffer, train_state)

            runner_state = (train_state, disc_train_state, env_state, last_obs, train_state_buffer, rng)
            return runner_state, (metric, validation_metrics)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, disc_train_state, env_state, obsv, train_state_buffer, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config.num_updates
        )

        agent_state = cls._agent_state(train_state=runner_state[0], disc_train_state=runner_state[1])

        return {"agent_state": agent_state,
                "training_metrics": metrics[0],
                "validation_metrics": metrics[1]}

    @classmethod
    def _predict_rewards(cls, inputs, discriminator, disc_train_state):
        logits, _ = discriminator.apply({'params': disc_train_state.params,
                                         'run_stats': disc_train_state.run_stats},
                                        *inputs, mutable=["run_stats"])

        plcy_prob = nn.sigmoid(logits)
        reward = jnp.squeeze(-jnp.log(1 - plcy_prob + 1e-6))

        return reward

    @classmethod
    def _discriminator_loss(cls, config, logits, targets):

        # binary cross entropy loss
        log_p = jax.nn.log_sigmoid(logits)
        log_not_p = jax.nn.log_sigmoid(-logits)
        bce_loss = jnp.mean(-targets * log_p - (1. - targets) * log_not_p)

        # bernoulli entropy
        discrim_prob = nn.sigmoid(logits)
        bernoulli_ent = (config.disc_ent_coef *
                         jnp.mean((1. - discrim_prob) * logits - nn.log_sigmoid(logits)))

        total_loss = bce_loss - bernoulli_ent

        return total_loss, discrim_prob


    @classmethod
    def _discriminator_gp(cls, config, discriminator, params, disc_train_state, inputs, rng):

        obs, act, obs_n, absorbing, log_prob = inputs
        # alpha = config.gp_alpha
        alpha = jax.random.uniform(rng)

        plcy_obs, demo_obs = jnp.split(obs, 2, axis=0)
        plcy_act, demo_act = jnp.split(act, 2, axis=0)
        plcy_obs_n, demo_obs_n = jnp.split(obs_n, 2, axis=0)
        plcy_absorbing, demo_absorbing = jnp.split(absorbing, 2, axis=0)
        _, log_prob = jnp.split(log_prob, 2, axis=0)

        interpolated_obs = alpha * demo_obs + (1 - alpha) * plcy_obs
        interpolated_act = alpha * demo_act + (1 - alpha) * plcy_act
        interpolated_obs_n = alpha * demo_obs_n + (1 - alpha) * plcy_obs_n
        interpolated_absorbing = 0.0 * demo_absorbing # assume interpolated state to be non-absorbing 
        interpolated_log_prob = jnp.ones_like(log_prob) # assume interpolated state to be non-absorbing 

        grad_obs, grad_act, grad_obs_n = jax.grad(lambda s, a, sn, ab, logp: jnp.mean(discriminator.apply({'params': params,
                                                'run_stats': disc_train_state.run_stats}, s, a, sn, ab, logp,
                                                mutable=["run_stats"])[0]), argnums=(0, 1, 2))(interpolated_obs, interpolated_act, interpolated_obs_n, interpolated_absorbing, interpolated_log_prob)

        if config.reward_type == "state-action":
            grad_norm = jnp.sqrt(jnp.sum(jnp.square(grad_obs)) + jnp.sum(jnp.square(grad_act)))
        elif config.reward_type in ["state-based", "shaped"]:
            grad_norm = jnp.sqrt(jnp.sum(jnp.square(grad_obs)) + jnp.sum(jnp.square(grad_obs_n)))
        elif config.reward_type == "shaped-sa":
            grad_norm = jnp.sqrt(jnp.sum(jnp.square(grad_obs)) + jnp.sum(jnp.square(grad_obs_n)) + jnp.sum(jnp.square(grad_act)))

        gp_loss = (grad_norm - 1.0) ** 2

        return gp_loss



    @classmethod
    def _get_discriminator_targets(cls, plcy_batch_size, expert_batch_size):
        plcy_target = jnp.zeros(shape=(plcy_batch_size,))
        expert_target = jnp.ones(shape=(expert_batch_size,))
        return plcy_target, expert_target

    @classmethod
    def play_policy(cls, env,
                    agent_conf: GAILAgentConf,
                    agent_state: GAILAgentState,
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
                           agent_conf: GAILAgentConf,
                           agent_state: GAILAgentState,
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