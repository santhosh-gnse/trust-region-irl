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
from loco_mujoco.algorithms import (FullyConnectedNet, Transition, TrainState,
                                    TrainStateBuffer, MetricHandlerTransition)
from loco_mujoco.core.wrappers import LogWrapper, NStepWrapper, LogEnvState, VecEnv, NormalizeVecReward, SummaryMetrics
from loco_mujoco.utils import MetricsHandler, ValidationSummary
from loco_mujoco.trajectory import TrajectoryTransitions, Trajectory
from trust_region_irl.algorithms.trirl_ppo.flax_loco_mjx.policy import ActorCritic
from trust_region_irl.algorithms.trirl_ppo.flax_loco_mjx.discriminator import get_discriminator, get_reward_approximator
import os
from trust_region_irl.algorithms.trirl_ppo.flax_loco_mjx.general_properties import GeneralProperties
import logging
from typing import NamedTuple
from flax.core import FrozenDict
import time
from pathlib import Path
import pickle
from trust_region_irl.algorithms.trirl_ppo.flax_loco_mjx.reward_correction import make_chunked_ensemble_rew_correct
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

@struct.dataclass
class ParamsBuffer:
    params: FrozenDict
    n: jax.Array  # index to add new entries at
    size: jax.Array # buffer size

    def serialize(self):
        serialized_params = flax.serialization.to_state_dict(self.params)
        serialized_n = np.asarray(jax.device_get(self.n))
        serialized_size = np.asarray(jax.device_get(self.size))

        return {
            "params": serialized_params,
            "n": serialized_n,
            "size": serialized_size,
        }


    @classmethod
    def create(cls, params: FrozenDict, size: int):
        return ParamsBuffer(
            params=jax.tree.map(lambda x: jnp.stack([x] * size), params),
            n=jnp.asarray(0, dtype=jnp.int32),
            size=size
        )

    @classmethod
    def add(cls, params_buffer, params: FrozenDict):
        index = params_buffer.n
        size = params_buffer.size

        def _update_buffer(buffer, new):
            def _full_size(_):
                new_last = jnp.expand_dims(new, axis=0)
                shifted = jnp.concatenate([buffer[1:], new_last], axis=0)
                return shifted
            
            def _not_full_size(_):
                return buffer.at[index].set(new)
            
            return jax.lax.cond(index < size, _not_full_size, _full_size, operand=None)

        params_updated = jax.tree.map(_update_buffer, params_buffer.params, params)    
        return params_buffer.replace(
            params=params_updated,
            n=jnp.minimum(index + 1, size),
        )

    @classmethod
    def sample(cls, params_buffer):
        level = jnp.minimum(params_buffer.n, params_buffer.size)
        return params_buffer.params, level

    @classmethod
    def sample_oldest(cls, params_buffer, k=1):
        oldest_params = jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, start_index=0, slice_size=k),
            params_buffer.params
        )

        if k == 1:
            oldest_params = jax.tree.map(lambda x: x[0], oldest_params)
            
        return oldest_params


@struct.dataclass
class EtasBuffer:
    etas: jax.Array
    n: jax.Array  # index to add new entries at
    size: jax.Array # buffer size

    def serialize(self):
        serialized_etas = np.asarray(jax.device_get(self.etas))
        serialized_n = np.asarray(jax.device_get(self.n))
        serialized_size = np.asarray(jax.device_get(self.size))

        return {
            "etas": serialized_etas,
            "n": serialized_n,
            "size": serialized_size,
        }

    @classmethod
    def create(cls, etas: jax.Array, size: int):
        return EtasBuffer(
            etas=jax.tree.map(lambda x: jnp.stack([jnp.zeros_like(x)] * size), etas),
            n=jnp.asarray(0, dtype=jnp.int32),
            size=size
        )

    @classmethod
    def add(cls, etas_buffer, etas: jax.Array):
        index = etas_buffer.n
        size = etas_buffer.size

        def _update_buffer(buffer, new):
            def _full_size(_):
                new_last = jnp.expand_dims(new, axis=0)
                shifted = jnp.concatenate([buffer[1:], new_last], axis=0)
                return shifted
            
            def _not_full_size(_):
                return buffer.at[index].set(new)
            
            return jax.lax.cond(index < size, _not_full_size, _full_size, operand=None)

        etas_updated   = jax.tree.map(_update_buffer, etas_buffer.etas, etas)        
        return etas_buffer.replace(
            etas=etas_updated,
            n=jnp.minimum(index + 1, size),
        )

    @classmethod
    def sample(cls, etas_buffer):
        n = jnp.minimum(etas_buffer.n, etas_buffer.size)
        etas = etas_buffer.etas
        s = etas.shape[0]
        i = jnp.arange(s, dtype=jnp.int32)
        dest = i + (i >= n).astype(jnp.int32)

        out = jnp.zeros((s + 1,) + etas.shape[1:], dtype=etas.dtype)
        out = out.at[dest].set(etas)

        return out, n



class TRIRLTransition(NamedTuple):
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
    action_mean: jax.Array
    action_logstd: jax.Array
    corr_reward: jax.Array = None


@struct.dataclass
class TRIRLSummaryMetrics(SummaryMetrics):
    discriminator_output_policy: float = 0.0
    discriminator_output_expert: float = 0.0
    discriminator_gp_loss: float = 0.0


@dataclass(frozen=True)
class TRIRLAgentConf(AgentConfBase):
    config: DictConfig
    network: ActorCritic
    discriminator: Any
    reward_approximator: Any
    tx: Any
    disc_tx: Any
    reward_approximator_tx: Any
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
                "reward_approximator": flax.serialization.to_state_dict(self.reward_approximator),
                "expert_dataset": None} # never save dataset
        return _all

    @classmethod
    def from_dict(cls, d):
        config = OmegaConf.create(d["config"])
        tx, disc_tx, reward_approximator_tx = TRIRL_PPO._get_optimizer(config)
        return cls(config=config,
                   network=flax.serialization.from_state_dict(ActorCritic, d["network"]),
                   discriminator=flax.serialization.from_state_dict(get_discriminator(config.reward_type), d["discriminator"]),
                   reward_approximator=flax.serialization.from_state_dict(get_reward_approximator(config.reward_approximator_type), d["reward_approximator"]),
                   tx=tx, disc_tx=disc_tx, reward_approximator_tx=reward_approximator_tx,
                   expert_dataset=flax.serialization.from_state_dict(TrajectoryTransitions, d["expert_dataset"]))


@struct.dataclass
class TRIRLAgentState(AgentStateBase):
    train_state: TrainState
    disc_train_state: TrainState
    reward_approximator_train_state: TrainState

    def serialize(self):
        serialized_train_state = flax.serialization.to_state_dict(self.train_state)
        serialized_discriminator = flax.serialization.to_state_dict(self.disc_train_state)
        serialized_reward_approximator = flax.serialization.to_state_dict(self.reward_approximator_train_state)
        return {"train_state": serialized_train_state,
                "discriminator": serialized_discriminator,
                "reward_approximator": serialized_reward_approximator}

    @classmethod
    def from_dict(cls, d, agent_conf):
        train_state = TrainState(apply_fn=agent_conf.network, tx=agent_conf.tx, **d["train_state"])
        disc_state = TrainState(apply_fn=agent_conf.discriminator, tx=agent_conf.disc_tx, **d["discriminator"])
        reward_approximator_state = TrainState(apply_fn=agent_conf.reward_approximator, tx=agent_conf.reward_approximator_tx, **d["reward_approximator"])
        return cls(train_state, disc_state, reward_approximator_state)


class TRIRL_PPO(JaxRLAlgorithmBase):

    _agent_conf = TRIRLAgentConf
    _agent_state = TRIRLAgentState


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

        merged["beta"] = 1 / merged["ent_coef"]

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
            assert config.reward_approximator_type != "state-action", "Mocap data does not contain actions! Please choose a reward approximator type that does not use actions"

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
        model = TRIRL_PPO(config, env, run_path, writer, skip_init=True)

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
        use_mujoco = False

        env._viewer_params.pop("visualize_goal")
        if use_mujoco:
            self.play_policy_mujoco(env, agent_conf, agent_state, deterministic=deterministic, n_steps=n_steps, record=record)
        else:
            self.play_policy(env, agent_conf, agent_state, deterministic=deterministic, n_steps=n_steps, n_envs=n_envs, record=record)


    def train(self):
        out = self.train_fn(self.key)

        if self.save_model: 
            os.makedirs(self.save_path)

            agent_state = out["agent_state"]
            buffers = out["buffers"]

            path = Path(self.save_path)
            path = path / (self.__class__.__name__ + "_saved")
            path = path.with_suffix(self._saved_agent_suffix)
            
            # serialize agent state
            serialized_state = self.serialize(self.cached_agent_conf, agent_state)

            # serialize buffers
            serialized_buffers = self.serialize_buffers(buffers)

            # save
            serialized_agent = serialized_state | serialized_buffers
            with open(path, 'wb') as file:
                pickle.dump(serialized_agent, file)

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
        reward_approximator = get_reward_approximator(config.reward_approximator_type)(activation=config.activation,
                                          hidden_layer_dims=config.hidden_layers,
                                          output_dim=1, output_activation=None,
                                          use_running_mean_stand=True,
                                          squeeze_output=True,
                                          gamma=config.gamma)

        # set up optimizers
        tx, disc_tx, reward_approximator_tx = cls._get_optimizer(config)

        return cls._agent_conf(config, network, discriminator, reward_approximator, tx, disc_tx, reward_approximator_tx)

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

        reward_approximator_tx = optax.chain(
            optax.clip_by_global_norm(config.lr),
            optax.adamw(config.reward_approximator_lr, weight_decay=config.weight_decay, eps=1e-5),
        )


        return tx, disc_tx, reward_approximator_tx

    @classmethod
    def _train_fn(cls, rng, env, writer,
                  agent_conf: TRIRLAgentConf,
                  agent_state: TRIRLAgentState = None,
                  mh: MetricsHandler = None):

        # extract static agent info
        config, network, discriminator, reward_approximator, tx, disc_tx, reward_approximator_tx, expert_dataset =\
            (agent_conf.config, agent_conf.network,
             agent_conf.discriminator, agent_conf.reward_approximator, agent_conf.tx, agent_conf.disc_tx, agent_conf.reward_approximator_tx, agent_conf.expert_dataset)

        # extract current agent state
        if agent_state is not None:
            train_state, disc_train_state, reward_approximator_train_state = agent_state.train_state, agent_state.disc_train_state, agent_state.reward_approximator_train_state
        else:
            train_state, disc_train_state, reward_approximator_train_state = None, None, None

        if train_state is None:
            rng, _rng1, _rng2, _rng3 = jax.random.split(rng, 4)
            init_x = jnp.zeros(env.info.observation_space.shape)
            init_xn = jnp.zeros(env.info.observation_space.shape)
            init_a = jnp.zeros(env.info.action_space.shape)
            init_abs = jnp.array([0.0])
            network_params = network.init(_rng1, init_x)
            discrim_params = discriminator.init(_rng2, init_x, init_a, init_xn, init_abs)
            reward_approximator_params = reward_approximator.init(_rng3,  init_x, init_a, init_xn, init_abs)
        else:
            network_params = None
            discrim_params = None
            reward_approximator_params = None

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

        reward_approximator_train_state = TrainState.create(
            apply_fn=reward_approximator.apply,
            params=reward_approximator_params["params"] if reward_approximator_train_state is None else reward_approximator_train_state.params,
            run_stats=reward_approximator_params["run_stats"] if reward_approximator_train_state is None else reward_approximator_train_state.run_stats,
            tx=reward_approximator_tx,
        )
        etas = cls._eta_schedule(config, 0, config.total_timesteps) * jnp.ones((config.num_steps, config.num_envs))
        env = cls._wrap_env(env, config)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs)
        obsv, env_state = env.reset(reset_rng)

        train_state_buffer = TrainStateBuffer.create(train_state, config.validation_num)
        disc_buffer = ParamsBuffer.create(disc_train_state, config.disc_buffer_capacity)
        etas_buffer = EtasBuffer.create(etas, config.disc_buffer_capacity-1)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, disc_train_state, reward_approximator_train_state, env_state, last_obs, train_state_buffer, disc_buffer, etas_buffer, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                y, updates = network.apply({'params': train_state.params,
                                                  'run_stats': train_state.run_stats},
                                                 last_obs, mutable=["run_stats"])
                pi, value, action_mean, action_logstd = y
                train_state = train_state.replace(run_stats=updates['run_stats'])   # update stats
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                action_mean, action_logstd = jax.lax.stop_gradient(action_mean), jax.lax.stop_gradient(action_logstd)
                action_logstd = jnp.repeat(action_logstd[None, :], action_mean.shape[0], axis=0)

                # STEP ENV
                obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)

                # GET METRICS
                log_env_state = env_state.find(LogEnvState)
                logged_metrics = log_env_state.metrics

                transition = TRIRLTransition(
                    done, absorbing, action, value, reward, log_prob, last_obs, obsv, info, env_state.additional_carry.traj_state,
                    logged_metrics, action_mean, action_logstd, None
                )
                runner_state = (train_state, disc_train_state, reward_approximator_train_state, env_state, obsv, train_state_buffer, disc_buffer, etas_buffer, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            train_state, disc_train_state, reward_approximator_train_state, env_state, last_obs, train_state_buffer, disc_buffer, etas_buffer, rng = runner_state


            def _update_discriminator(runner_state, unused):
                disc_train_state, traj_batch, rng = runner_state
                def _get_one_batch(data, batch_size, rng):
                    data0, data1, data2, data3 = data
                    idx = jax.random.randint(rng, shape=(batch_size,), minval=0, maxval=data0.shape[0])
                    return data0[idx], data1[idx], data2[idx], data3[idx]

                def _discrim_loss(params, disc_train_state, inputs, targets, rng):
                    # obs, act, obs_n, absorbing = inputs
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
                rng, _rng1, _rng2, _rng3 = jax.random.split(rng, 4)
                batch_size = config.disc_minibatch_size

                obs = traj_batch.obs.reshape((-1, traj_batch.obs.shape[-1]))
                obs_n = traj_batch.obs_n.reshape((-1, traj_batch.obs_n.shape[-1]))
                act = traj_batch.action.reshape((-1, traj_batch.action.shape[-1]))
                absorbing = traj_batch.absorbing.flatten()

                obs_batch, obs_n_batch, act_batch, absorbing_batch = _get_one_batch((obs, obs_n, act, absorbing), batch_size, _rng1)
                demo_obs_batch, demo_obs_n_batch, demo_act_batch, demo_absorbing_batch = _get_one_batch((expert_dataset.observations, 
                                                                                                        expert_dataset.next_observations, 
                                                                                                        expert_dataset.actions, 
                                                                                                        expert_dataset.absorbings
                                                                                                        ), 
                                                                                                        batch_size, _rng2)
                
                # Create labels
                plcy_target = jnp.zeros(shape=(obs_batch.shape[0],))
                demo_target = jnp.ones(shape=(demo_obs_batch.shape[0],))

                # concatenate inputs and targets
                inputs = (jnp.concatenate([obs_batch, demo_obs_batch], axis=0),
                        jnp.concatenate([act_batch, demo_act_batch], axis=0),
                        jnp.concatenate([obs_n_batch, demo_obs_n_batch], axis=0),
                        jnp.concatenate([absorbing_batch, demo_absorbing_batch], axis=0)
                        )

                targets = jnp.concatenate([plcy_target, demo_target], axis=0)

                # update discriminator
                grad_fn = jax.value_and_grad(_discrim_loss, has_aux=True)
                (total_loss, (disc_train_state, discrim_probs_policy, discrim_probs_exp, discrim_gp_loss)), grads =\
                    grad_fn(disc_train_state.params, disc_train_state, inputs, targets, _rng3)

                # apply discriminator gradients
                disc_train_state = disc_train_state.apply_gradients(grads=grads)

                runner_state = (disc_train_state, traj_batch, rng)
                return runner_state, (total_loss, discrim_probs_policy, discrim_probs_exp, discrim_gp_loss)

            (disc_train_state, traj_batch, rng), (discrim_loss_info, discrim_probs_policy, discrim_probs_exp, discrim_gp_loss) = jax.lax.scan(
                _update_discriminator, (disc_train_state, traj_batch, rng), xs=None, length=config.n_disc_epochs
            )

            """ Reward Correction """
            chunked_correct = make_chunked_ensemble_rew_correct(
                cls._vmap_get_log_density_ratio,
                nr_steps=config.num_steps,
                nr_envs=config.num_envs,
                epsilon=config.epsilon,
                beta=config.beta,
                entropy_coef=config.ent_coef,
                maximum_eta=True
            )

            # Add new discriminator params to the buffer
            disc_buffer = ParamsBuffer.add(disc_buffer, disc_train_state)
            disc_params_sampled, level = ParamsBuffer.sample(disc_buffer) # get the whole buffer, only use till level during correction
            etas_sampled, _ = EtasBuffer.sample(etas_buffer) # etas are zero padded during correction

            corr_reward = chunked_correct(
                disc_params_sampled,
                discriminator,
                (traj_batch.obs.reshape((-1, traj_batch.obs.shape[-1])),
                traj_batch.action.reshape((-1, traj_batch.action.shape[-1])),
                traj_batch.obs_n.reshape((-1, traj_batch.obs_n.shape[-1])),
                traj_batch.absorbing.flatten(),
                ),
                etas_sampled,
                level,
                chunk_size=config.chunk_size,
            ).reshape(traj_batch.reward.shape)

            if config.handle_absorbing_states:
                corr_reward_absorbing_state = chunked_correct(
                    disc_params_sampled,
                    discriminator,
                    (traj_batch.obs_n.reshape((-1, traj_batch.obs_n.shape[-1])), # next state
                    0.0*traj_batch.action.reshape((-1, traj_batch.action.shape[-1])), # next actions don't matter in absorbing states
                    traj_batch.obs_n.reshape((-1, traj_batch.obs_n.shape[-1])), # next next states are the same as next states if in absorbing state
                    jnp.ones_like(traj_batch.absorbing.flatten()), # absorbing is true
                    ),
                    etas_sampled,
                    level,
                    chunk_size=config.chunk_size,
                ).reshape(traj_batch.reward.shape)
            else:
                corr_reward_absorbing_state = jnp.asarray(0.0)

            traj_batch = TRIRLTransition(
                    traj_batch.done, traj_batch.absorbing, traj_batch.action, traj_batch.value, traj_batch.reward, traj_batch.log_prob, traj_batch.obs, traj_batch.obs_n, traj_batch.info, traj_batch.traj_state,
                    traj_batch.metrics, traj_batch.action_mean, traj_batch.action_logstd, corr_reward
                )

            """ Fitting Corr Rewards """
            def _update_reward_fn_approximator(runner_state, unused):
                reward_approximator_train_state, traj_batch, rng = runner_state
                def _get_one_batch(data, batch_size, rng):
                    data0, data1, data2, data3, data4 = data
                    idx = jax.random.randint(rng, shape=(batch_size,), minval=0, maxval=data0.shape[0])
                    return data0[idx], data1[idx], data2[idx], data3[idx], data4[idx]

                def _approximator_loss(params, reward_approximator_train_state, inputs, targets, rng):
                    logits, updates = reward_approximator.apply({'params': params,
                                                            'run_stats': reward_approximator_train_state.run_stats},
                                                            *inputs, mutable=["run_stats"])

                    # update running statistics
                    reward_approximator_train_state = reward_approximator_train_state.replace(run_stats=updates["run_stats"])

                    # calculate loss
                    total_loss = jnp.mean(jnp.square(logits - targets))

                    return total_loss, (reward_approximator_train_state)

                # Get one batch of policy and expert demonstrations
                rng, _rng1, _rng2 = jax.random.split(rng, 3)
                batch_size = config.disc_minibatch_size

                obs = traj_batch.obs.reshape((-1, traj_batch.obs.shape[-1]))
                obs_n = traj_batch.obs_n.reshape((-1, traj_batch.obs_n.shape[-1]))
                act = traj_batch.action.reshape((-1, traj_batch.action.shape[-1]))
                absorbing = traj_batch.absorbing.flatten()
                target = traj_batch.corr_reward.flatten()

                obs_batch, obs_n_batch, act_batch, absorbing_batch, target_batch = _get_one_batch((obs, obs_n, act, absorbing, target), batch_size, _rng1)
                
                # concatenate inputs and targets
                inputs = (obs_batch,
                        act_batch,
                        obs_n_batch,
                        absorbing_batch
                        )

                targets = target_batch

                # update approximator
                grad_fn = jax.value_and_grad(_approximator_loss, has_aux=True)
                (total_loss, (reward_approximator_train_state)), grads =\
                    grad_fn(reward_approximator_train_state.params, reward_approximator_train_state, inputs, targets, _rng2)

                # apply gradients
                reward_approximator_train_state = reward_approximator_train_state.apply_gradients(grads=grads)

                runner_state = (reward_approximator_train_state, traj_batch, rng)
                return runner_state, (total_loss)

            if config.reward_fn_approximator:
                (reward_approximator_train_state, traj_batch, rng), (approximator_loss_info) = jax.lax.scan(
                    _update_reward_fn_approximator, (reward_approximator_train_state, traj_batch, rng), xs=None, length=config.n_reward_approximator_epochs
                )

                fitted_corr_reward = cls._get_reward_approximator_prediction((traj_batch.obs.reshape((-1, traj_batch.obs.shape[-1])),
                                                                    traj_batch.action.reshape((-1, traj_batch.action.shape[-1])),
                                                                    traj_batch.obs_n.reshape((-1, traj_batch.obs_n.shape[-1])),
                                                                    traj_batch.absorbing.flatten(),
                                                                    ),
                                                                    reward_approximator, reward_approximator_train_state).reshape(traj_batch.reward.shape)

                fitted_corr_reward_absorbing_state = cls._get_reward_approximator_prediction((traj_batch.obs_n.reshape((-1, traj_batch.obs_n.shape[-1])), # next state
                                                                    0.0*traj_batch.action.reshape((-1, traj_batch.action.shape[-1])), # next actions don't matter in absorbing states
                                                                    traj_batch.obs_n.reshape((-1, traj_batch.obs_n.shape[-1])), # next next states are the same as next states if in absorbing state
                                                                    jnp.ones_like(traj_batch.absorbing.flatten()), # absorbing is true
                                                                    ),
                                                                    reward_approximator, reward_approximator_train_state).reshape(traj_batch.reward.shape)

                traj_batch = TRIRLTransition(
                        traj_batch.done, traj_batch.absorbing, traj_batch.action, traj_batch.value, traj_batch.reward, traj_batch.log_prob, traj_batch.obs, traj_batch.obs_n, traj_batch.info, traj_batch.traj_state,
                        traj_batch.metrics, traj_batch.action_mean, traj_batch.action_logstd, fitted_corr_reward
                    )
                corr_reward_absorbing_state = fitted_corr_reward_absorbing_state
            else:
                approximator_loss_info = jnp.asarray(0.0)


            # CALCULATE ADVANTAGE
            def _calculate_gae(transition, rewards_next_state, train_state):

                done, absorbing, value, reward, corr_reward, obs, obs_n = (
                    transition.done,
                    transition.absorbing,
                    transition.value,
                    transition.reward,
                    transition.corr_reward,
                    transition.obs,
                    transition.obs_n
                )

                # compute proportion of each reward
                reward = (config.proportion_env_reward * reward +
                            (1 - config.proportion_env_reward) * corr_reward)
                H_terminal = jnp.sum(jnp.log(env.mdp_info.action_space.high - env.mdp_info.action_space.low))

                y, _ = network.apply({'params': train_state.params,
                                    'run_stats': train_state.run_stats},
                                    obs_n, mutable=["run_stats"])
                _, next_value, _, _ = y

                if config.handle_absorbing_states:
                    terminal_tail = (config.gamma / (1.0 - config.gamma)) * (rewards_next_state + config.ent_coef * H_terminal)
                    delta = reward + config.gamma * next_value * (1.0 - absorbing) + (absorbing * terminal_tail) - value
                else:
                    delta = reward + config.gamma * next_value * (1.0 - absorbing) - value
    
                init_advantages = delta[-1]

                def _get_advantages(carry, t):
                    prev_advantage = carry[0]
                    advantage = delta[t] + config.gamma * config.gae_lambda * (1 - done[t]) * prev_advantage
                    return (advantage,), advantage

                _, advantages = jax.lax.scan(_get_advantages, (init_advantages,), jnp.arange(config.num_steps - 2, -1, -1))
                advantages = jnp.concatenate([advantages[::-1], jnp.array([init_advantages])])
                returns = advantages + value
                return advantages, returns

            advantages, targets = _calculate_gae(traj_batch, corr_reward_absorbing_state, train_state)

            # Compute new scheduled etas
            eta = cls._eta_schedule(config, jnp.max(traj_batch.metrics.timestep * config.num_envs), config.total_timesteps)
            etas_buffer = EtasBuffer.add(etas_buffer, eta * jnp.ones((config.num_steps, config.num_envs)))

            # UPDATE ACTOR & CRITIC NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        y, _ = network.apply({'params': params, 'run_stats': train_state.run_stats},
                                             traj_batch.obs, mutable=["run_stats"])
                        pi, value, action_mean, action_logstd = y
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
                        entropy = config.ent_coef * entropy # scaling to reduce critic loss

                        # TRUST REGION PENALTY
                        tr_loss_maha = 0.5 * jnp.sum(((traj_batch.action_mean - action_mean)/jnp.exp(traj_batch.action_logstd)) ** 2, axis=1)
                        tr_loss_cov_part = 0.5 * jnp.sum(2.0 * (traj_batch.action_logstd - action_logstd) + (jnp.exp(action_logstd)/jnp.exp(traj_batch.action_logstd))**2 - 1.0, axis=1)
                        trust_region_loss = tr_loss_maha + tr_loss_cov_part
                        trust_region_loss = trust_region_loss.mean()

                        total_loss = (
                            loss_actor
                            + config.vf_coef * value_loss
                            - config.ent_coef * entropy
                            + eta * trust_region_loss
                        )
                        return total_loss, (value_loss, loss_actor, trust_region_loss, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, disc_train_state, reward_approximator_train_state, traj_batch, advantages, targets, rng = update_state
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
                update_state = (train_state, disc_train_state, reward_approximator_train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, disc_train_state, reward_approximator_train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )
            train_state = update_state[0]
            rng = update_state[-1]

            counter = ((train_state.step + 1) // config.num_minibatches) // config.update_epochs

            logged_metrics = traj_batch.metrics
            logging_corr_reward = traj_batch.corr_reward
            metric = TRIRLSummaryMetrics(
                discriminator_output_policy=jnp.mean(discrim_probs_policy),
                discriminator_output_expert=jnp.mean(discrim_probs_exp),
                discriminator_gp_loss=jnp.mean(discrim_gp_loss),
                mean_episode_return=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 0.0)) / jnp.sum(logged_metrics.done),
                mean_episode_length=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 0.0)) / jnp.sum(logged_metrics.done),
                max_timestep=jnp.max(logged_metrics.timestep * config.num_envs),
            )

            def _evaluation_step():

                def _eval_env(runner_state, unused):
                    train_state, disc_train_state, reward_approximator_train_state, env_state, last_obs, train_state_buffer, disc_buffer, etas_buffer, rng = runner_state

                    # SELECT ACTION
                    rng, _rng = jax.random.split(rng)
                    y, updates = train_state.apply_fn({'params': train_state.params,
                                                       'run_stats': train_state.run_stats},
                                                      last_obs, mutable=["run_stats"])
                    pi, value, _, _ = y
                    train_state = train_state.replace(run_stats=updates['run_stats'])  # update stats
                    action = pi.sample(seed=_rng)

                    # STEP ENV
                    obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)

                    # GET METRICS
                    log_env_state = env_state.find(LogEnvState)
                    logged_metrics = log_env_state.metrics

                    transition = MetricHandlerTransition(env_state, logged_metrics)

                    runner_state = (train_state, disc_train_state, reward_approximator_train_state, env_state, obsv, train_state_buffer, disc_buffer, etas_buffer, rng)
                    return runner_state, transition

                rng = runner_state[-1]
                reset_rng = jax.random.split(rng, config.validation_num_envs)
                obsv, env_state = env.reset(reset_rng)
                runner_state_eval = (train_state, disc_train_state, reward_approximator_train_state, env_state, obsv, train_state_buffer, disc_buffer, etas_buffer, rng)

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

                metric, logged_metrics, logging_corr_reward, loss_info, log_std, discrim_loss, approximator_loss, misc = metric
                total_policy_loss, (value_loss, loss_actor, loss_trust_region, entropy) = loss_info
                eta, level = misc
                log_step = metric.max_timestep.astype('int32')
                log_dict = {
                    "rollout/episode_return": metric.mean_episode_return,
                    "rollout/episode_length": metric.mean_episode_length,
                    "rollout/dones": jnp.mean(logged_metrics.done),
                    "rollout/corr_reward" : jnp.mean(logging_corr_reward),
                    "loss/total_policy_loss": jnp.mean(total_policy_loss),
                    "loss/policy_loss": jnp.mean(loss_actor),
                    "loss/critic_loss": jnp.mean(value_loss),
                    "loss/discriminator_gp": metric.discriminator_gp_loss,
                    "loss/discriminator_loss": jnp.mean(discrim_loss),
                    "loss/trust_region_loss": jnp.mean(loss_trust_region),
                    "loss/rew_approx_loss": jnp.mean(approximator_loss),
                    "policy/entropy": jnp.mean(entropy),
                    "policy/std": jnp.mean(jnp.exp(log_std)),
                    "discriminator/policy_pred": metric.discriminator_output_policy,
                    "discriminator/expert_pred": metric.discriminator_output_expert,
                    "rollout/nr_env_steps": log_step,
                    "correction/eta": eta,
                    # "correction/time": correction_time,
                    "correction/buffer_size": level,
                    } 
                for name, value in log_dict.items():
                    if isinstance(value, jax.Array):
                        value = value.__array__()
                    writer.add_scalar(name, value, log_step.__array__())
                    if wandb.run is not None:
                        wandb.log({name: value, "global_step": int(log_step)}, commit=False)
                    rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │", flush=False)

                rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")

            jax.debug.callback(metrics_callback, (metric, logged_metrics, logging_corr_reward, loss_info, train_state.params["log_std"], discrim_loss_info, approximator_loss_info, (eta, level)))


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

            runner_state = (train_state, disc_train_state, reward_approximator_train_state, env_state, last_obs, train_state_buffer, disc_buffer, etas_buffer, rng)
            return runner_state, (metric, validation_metrics)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, disc_train_state, reward_approximator_train_state, env_state, obsv, train_state_buffer, disc_buffer, etas_buffer, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config.num_updates
        )

        agent_state = cls._agent_state(train_state=runner_state[0], disc_train_state=runner_state[1], reward_approximator_train_state=runner_state[2])

        return {"agent_state": agent_state,
                "buffers": {"disc_buffer": runner_state[6], "etas_buffer": runner_state[7]},
                "training_metrics": metrics[0],
                "validation_metrics": metrics[1]}


    @classmethod
    def _eta_schedule(cls, config, global_step, total_timesteps):
        init_eta = jnp.asarray(config.init_eta)
        def _linear_schedule_eta(global_step, total_timesteps):
            fraction = 1.0 - jnp.clip(global_step, min=None, max=total_timesteps) / total_timesteps
            return init_eta * fraction

        if config.const_eta:
            return init_eta
        else:
            return _linear_schedule_eta(global_step, total_timesteps)

    @classmethod
    def _get_reward_approximator_prediction(cls, inputs, reward_approximator, reward_approximator_train_state):
        logits, _ = reward_approximator.apply({'params': reward_approximator_train_state.params,
                                         'run_stats': reward_approximator_train_state.run_stats},
                                        *inputs, mutable=["run_stats"])

        return logits

    @classmethod
    def _get_log_density_ratio(cls, inputs, discriminator, disc_train_state):
        logits, _ = discriminator.apply({'params': disc_train_state.params,
                                         'run_stats': disc_train_state.run_stats},
                                        *inputs, mutable=["run_stats"])

        return logits


    @classmethod
    def _vmap_get_log_density_ratio(cls, inputs, discriminator, disc_train_state):

        vmap_log_density_ratio = jax.vmap(cls._get_log_density_ratio, in_axes=(0, None, None), out_axes=0)
        logits = vmap_log_density_ratio(inputs, discriminator, disc_train_state)

        return logits


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

        obs, act, obs_n, absorbing = inputs
        alpha = jax.random.uniform(rng)

        plcy_obs, demo_obs = jnp.split(obs, 2, axis=0)
        plcy_act, demo_act = jnp.split(act, 2, axis=0)
        plcy_obs_n, demo_obs_n = jnp.split(obs_n, 2, axis=0)
        plcy_absorbing, demo_absorbing = jnp.split(absorbing, 2, axis=0)

        interpolated_obs = alpha * demo_obs + (1 - alpha) * plcy_obs
        interpolated_act = alpha * demo_act + (1 - alpha) * plcy_act
        interpolated_obs_n = alpha * demo_obs_n + (1 - alpha) * plcy_obs_n
        interpolated_absorbing = 0.0 * demo_absorbing # assume interpolated state to be non-absorbing 

        grad_obs, grad_act, grad_obs_n = jax.grad(lambda s, a, sn, ab: jnp.mean(discriminator.apply({'params': params,
                                                'run_stats': disc_train_state.run_stats}, s, a, sn, ab,
                                                mutable=["run_stats"])[0]), argnums=(0, 1, 2))(interpolated_obs, interpolated_act, interpolated_obs_n, interpolated_absorbing)

        if config.reward_type == "state-action":
            grad_norm = jnp.sqrt(jnp.sum(jnp.square(grad_obs)) + jnp.sum(jnp.square(grad_act)))
        elif config.reward_type in ["state-based", "shaped"]:
            grad_norm = jnp.sqrt(jnp.sum(jnp.square(grad_obs)) + jnp.sum(jnp.square(grad_obs_n))) 

        gp_loss = (grad_norm - 1.0) ** 2

        return gp_loss



    @classmethod
    def _get_discriminator_targets(cls, plcy_batch_size, expert_batch_size):
        plcy_target = jnp.zeros(shape=(plcy_batch_size,))
        expert_target = jnp.ones(shape=(expert_batch_size,))
        return plcy_target, expert_target

    @classmethod
    def play_policy(cls, env,
                    agent_conf: TRIRLAgentConf,
                    agent_state: TRIRLAgentState,
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
            pi, _, _, _ = y
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
                           agent_conf: TRIRLAgentConf,
                           agent_state: TRIRLAgentState,
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