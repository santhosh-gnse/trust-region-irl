import os
import ast
from omegaconf import open_dict
import warnings
from dataclasses import dataclass, replace
from typing import Any
from omegaconf import DictConfig, OmegaConf, ListConfig
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
import flax
import optax
import logging
import wandb
from functools import partial
import tree

from trust_region_irl.environments.loco_mjx.base_algorithm import JaxRLAlgorithmBase, AgentConfBase, AgentStateBase
from loco_mujoco.algorithms import (ActorCritic,
                                    Transition, TrainState, TrainStateBuffer, MetricHandlerTransition)
from loco_mujoco.core.wrappers import LogWrapper, NStepWrapper, LogEnvState, VecEnv, NormalizeVecReward, SummaryMetrics
from loco_mujoco.utils import MetricsHandler, ValidationSummary
from loco_mujoco import TaskFactory
from loco_mujoco.trajectory import TrajectoryTransitions, Trajectory

import os
from trust_region_irl.algorithms.near_ppo.flax_loco_mjx.energy_function import get_energyfn
from trust_region_irl.algorithms.near_ppo.flax_loco_mjx.general_properties import GeneralProperties
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


class NEARTransition(NamedTuple):
    done: jax.Array
    absorbing: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    energy_reward: jax.Array
    energy: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    obs_n: jax.Array
    info: jax.Array
    traj_state: "TrajState"
    metrics: "Metrics"


@dataclass(frozen=True)
class NEARAgentConf(AgentConfBase):
    config: DictConfig
    network: ActorCritic
    energyfn: Any
    tx: Any
    energyfn_tx: Any
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
                "energyfn": flax.serialization.to_state_dict(self.energyfn),
                "expert_dataset": None} # never save dataset
        return _all

    @classmethod
    def from_dict(cls, d):
        config = OmegaConf.create(d["config"])
        tx, energyfn_tx = NEAR_PPO._get_optimizer(config)
        return cls(config=config,
                   network=flax.serialization.from_state_dict(ActorCritic, d["network"]),
                   energyfn=flax.serialization.from_state_dict(get_energyfn(config.ncsnv1), d["energyfn"]),
                   tx=tx, energyfn_tx=energyfn_tx,
                   expert_dataset=flax.serialization.from_state_dict(TrajectoryTransitions, d["expert_dataset"]))


@struct.dataclass
class NEARAgentState(AgentStateBase):
    train_state: Any
    energyfn_train_state: Any

    def serialize(self):
        if self.train_state is not None:
            serialized_train_state = flax.serialization.to_state_dict(self.train_state)
        else:
            serialized_train_state = None
        serialized_energyfn = flax.serialization.to_state_dict(self.energyfn_train_state)
        return {"train_state": serialized_train_state,
                "energyfn": serialized_energyfn}

    @classmethod
    def from_dict(cls, d, agent_conf):
        train_state = TrainState(apply_fn=agent_conf.network, tx=agent_conf.tx, **d["train_state"])
        energyfn_state = TrainState(apply_fn=agent_conf.energyfn, tx=agent_conf.energyfn_tx, **d["energyfn"])
        return cls(train_state, energyfn_state)



class NEAR_PPO(JaxRLAlgorithmBase):

    _agent_conf = NEARAgentConf
    _agent_state = NEARAgentState

    def __init__(self, config, env, eval_env, run_path, writer, skip_init=False) -> None:

       if not skip_init:
            config, expert_dataset = self.prepare_config_and_expert_dataset(config, env)

            agent_conf = self.init_agent_conf(env, config)
            agent_conf = agent_conf.add_expert_dataset(expert_dataset)
            mh = MetricsHandler(config, env) if config.validation_active else None

            self.ncsn_train_fn = jax.jit(self.build_ncsn_train_fn(env, writer, agent_conf))

            train_fn = self.build_near_train_fn(env, writer, agent_conf, mh=mh)
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
            assert config.state_based == True, "Mocap data does not contain actions! Please choose the state based option"

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
        model = NEAR_PPO(config, env, run_path, writer, skip_init=True)

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

        if use_mujoco:
            self.play_policy_mujoco(env, agent_conf, agent_state, deterministic=deterministic, n_steps=n_steps, record=record)
        else:
            self.play_policy(env, agent_conf, agent_state, deterministic=deterministic, n_steps=n_steps, n_envs=n_envs, record=record)


    def train(self):
        ncsn_key, ppo_key, self.key = jax.random.split(self.key, 3)
        
        rlx_logger.info(f"Training Energy Based Reward")
        ncsn_out = self.ncsn_train_fn(ncsn_key)

        rlx_logger.info(f"Training PPO Policy")
        out = self.train_fn(ppo_key, ncsn_out)

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
    def build_ncsn_train_fn(cls, env, writer, agent_conf: AgentConfBase):
        """ Returns the NCSN train function. """
        return lambda rng_key: cls._ncsn_train_fn(rng_key, env, writer, agent_conf)


    @classmethod
    def build_near_train_fn(cls, env, writer, agent_conf: AgentConfBase, mh: MetricsHandler = None):
        """ Returns the PPO train function to optimize learnt rewards. """
        return lambda rng_key, agent_state: cls._train_fn(rng_key, env, writer, agent_conf, agent_state, mh=mh)

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
            config.num_updates_ncsn = (
                    config.total_samples_ncsn // config.nr_epochs_ncsn // config.batch_size_ncsn)
            if config.sigma_inference_ncsn == -1:
                config.annealing = True
                config.sigma_inference_ncsn = 0
            else:
                config.annealing = False

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
            actor_obs_ind=actor_obs_ind,
            critic_obs_ind=critic_obs_ind
        )

        energyfn = get_energyfn(config.ncsnv1)(encoder_hidden_layer_dims=config.hidden_layers_encoder_ncsn,
                                          decoder_hidden_layer_dims=config.hidden_layers_decoder_ncsn,
                                          use_running_mean_stand=config.use_running_mean_stand)

        # set up optimizers
        tx, energyfn_tx = cls._get_optimizer(config)

        return cls._agent_conf(config, network, energyfn, tx, energyfn_tx)

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

        tx = optax.apply_if_finite(tx, max_consecutive_errors=10000000)

        energyfn_tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adamw(config.ncsn_lr, weight_decay=config.weight_decay, eps=1e-5),
        )

        return tx, energyfn_tx


    @classmethod
    def _ncsn_train_fn(cls, rng, env, writer,
                  agent_conf: NEARAgentConf,
                  agent_state: NEARAgentState = None):

        """
        Setup train state and networks 
        """

        config, energyfn, energyfn_tx, expert_dataset =\
            (agent_conf.config, agent_conf.energyfn, agent_conf.energyfn_tx, agent_conf.expert_dataset)


        # extract current agent state
        if agent_state is not None:
            energyfn_train_state = agent_state.energyfn_train_state
        else:
            energyfn_train_state = None

        if energyfn_train_state is None:
            rng, _rng1 = jax.random.split(rng, 2)
            init_cond = jnp.array([0.0])

            if config.state_based:
                init_sample = jnp.concatenate([jnp.zeros(env.info.observation_space.shape), jnp.zeros(env.info.observation_space.shape)])
            else:
                init_sample = jnp.concatenate([jnp.zeros(env.info.observation_space.shape), jnp.zeros(env.info.action_space.shape)])
            energyfn_params = energyfn.init(_rng1, init_sample, init_cond)
        else:
            energyfn_params = None

        # If RMS is off 
        if "run_stats" not in energyfn_params:
            energyfn_params = dict(energyfn_params)
            energyfn_params["run_stats"] = {}

        # init new train states from old params
        energyfn_train_state = TrainState.create(
            apply_fn=energyfn.apply,
            params=energyfn_params["params"] if energyfn_train_state is None else energyfn_train_state.params,
            run_stats=energyfn_params["run_stats"] if energyfn_train_state is None else energyfn_train_state.run_stats,
            tx=energyfn_tx,
        )

        """
        Noise Conditioned Score Networks Update
        """

        # Flatten expert dataset to arrays (same semantics as RLX code)
        expert_states = expert_dataset.observations
        expert_next_states = expert_dataset.next_observations
        expert_actions = expert_dataset.actions
        if hasattr(expert_dataset, "absorbings"):
            expert_absorbing = expert_dataset.absorbings.flatten()
        elif hasattr(expert_dataset, "dones"):
            expert_absorbing = expert_dataset.dones.flatten()
        else:
            expert_absorbing = jnp.zeros(expert_states.shape[0], dtype=expert_states.dtype)

        nr_minibatches_ncsn = config.batch_size_ncsn // config.minibatch_size_ncsn

        def ncsn_learning_iteration(learning_iteration_carry, ncsn_learning_iteration_step):
            energyfn_train_state, \
            (expert_states, expert_actions, expert_next_states, expert_absorbing), \
            rng = learning_iteration_carry

            def ncsn_loss_fn(energyfn_params, expert_state, expert_action, expert_next_state, rng_single):
                """
                Denoising Score Matching
                """
                # Geometric schedule over sigmas
                rng_single, label_rng = jax.random.split(rng_single)
                sigmas = jnp.exp(jnp.linspace(jnp.log(config.sigma_begin_ncsn), jnp.log(config.sigma_end_ncsn), config.L_ncsn))
                conds = jnp.arange(config.L_ncsn)
                used_cond = jax.random.choice(label_rng, conds)
                used_sigma = sigmas[used_cond]

                # perturbing expert sample
                if config.state_based:
                    sample = jnp.concatenate([expert_state.flatten(), expert_next_state.flatten()])
                else:
                    sample = jnp.concatenate([expert_state.flatten(), expert_action.flatten()])
                perturbed_sample = sample + jax.random.normal(rng_single, shape=sample.shape) * used_sigma
                target = - 1.0 / (used_sigma ** 2) * (perturbed_sample - sample)

                def energy_sum(x, cond):
                    mutable = ['run_stats'] if config.use_running_mean_stand else []
                    energy, _ = energyfn.apply({"params": energyfn_params, "run_stats": energyfn_train_state.run_stats}, x, cond, mutable=mutable)
                    return jnp.sum(energy)

                if config.ncsnv1:
                    pred_score = jax.grad(energy_sum, argnums=0)(perturbed_sample, used_cond)
                else:
                    pred_score = jax.grad(energy_sum, argnums=0)(perturbed_sample, used_sigma)

                dsm_loss = jnp.mean(0.5 * jnp.sum((pred_score - target) ** 2) * (used_sigma ** config.anneal_power_ncsn))

                metrics = {
                    "loss/energyfn_loss": dsm_loss,
                }
                return dsm_loss, metrics

            # Shuffle expert data and take a batch of size batch_size_ncsn
            rng, shuffle_rng = jax.random.split(rng)
            perm = jax.random.permutation(shuffle_rng, expert_states.shape[0])
            expert_states = expert_states[perm]
            expert_actions = expert_actions[perm]
            expert_next_states = expert_next_states[perm]

            batch_expert_states = expert_states[:config.batch_size_ncsn]
            batch_expert_actions = expert_actions[:config.batch_size_ncsn]
            batch_expert_next_states = expert_next_states[:config.batch_size_ncsn]

            vmap_ncsn_loss_fn = jax.vmap(ncsn_loss_fn, in_axes=(None, 0, 0, 0, 0), out_axes=(0, 0))
            safe_mean = lambda x: jnp.mean(x) if x is not None else x
            mean_vmapped_ncsn_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_ncsn_loss_fn(*a, **k))
            grad_ncsn_loss_fn = jax.value_and_grad(mean_vmapped_ncsn_loss_fn, argnums=(0), has_aux=True)


            # Build shuffled minibatch indices for this iteration
            rng, subkey = jax.random.split(rng)
            batch_indices_ncsn = jnp.tile(jnp.arange(config.batch_size_ncsn), (config.nr_epochs_ncsn, 1))
            batch_indices_ncsn = jax.random.permutation(subkey, batch_indices_ncsn, axis=1, independent=True)
            batch_indices_ncsn = batch_indices_ncsn.reshape((config.nr_epochs_ncsn * nr_minibatches_ncsn, config.minibatch_size_ncsn))

            def ncsn_minibatch_update(carry, minibatch_indices_ncsn):
                energyfn_train_state, rng = carry

                rng, label_rng = jax.random.split(rng)
                mb_keys = jax.random.split(label_rng, config.minibatch_size_ncsn)

                # NCSN UPDATE
                (near_loss, metrics), grads = grad_ncsn_loss_fn(
                    energyfn_train_state.params,
                    batch_expert_states[minibatch_indices_ncsn],
                    batch_expert_actions[minibatch_indices_ncsn],
                    batch_expert_next_states[minibatch_indices_ncsn],
                    mb_keys,
                )

                energyfn_train_state = energyfn_train_state.apply_gradients(grads=grads)
                metrics["gradients/energyfn_grad_norm"] = optax.global_norm(grads)

                carry = (energyfn_train_state, rng)
                return carry, metrics

            init_carry = (energyfn_train_state, rng)
            (energyfn_train_state, rng), ncsn_optimization_metrics_seq = jax.lax.scan(ncsn_minibatch_update, init_carry, batch_indices_ncsn)

            ncsn_optimization_metrics = tree.map_structure(lambda x: jnp.mean(x, axis=0), ncsn_optimization_metrics_seq)
            combined_learning_iteration_step_ncsn = ncsn_learning_iteration_step + 1
            steps_per_iter = config.nr_epochs_ncsn * nr_minibatches_ncsn
            samples_per_iter = steps_per_iter * config.minibatch_size_ncsn
            steps_metrics = {
                "steps/nr_updates_ncsn": combined_learning_iteration_step_ncsn * steps_per_iter,
                "steps/nr_samples_ncsn": combined_learning_iteration_step_ncsn * samples_per_iter,
            }
            combined_metrics = {**ncsn_optimization_metrics, **steps_metrics}

            def metrics_callback(metric):
                optimization_metrics_ncsn, nr_samples_ncsn = metric
                rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐", flush=False)

                log_dict = {
                    "loss/energyfn_loss": optimization_metrics_ncsn["loss/energyfn_loss"],
                    "gradients/energyfn_grad_norm": optimization_metrics_ncsn["gradients/energyfn_grad_norm"],
                    "nr_samples_ncsn": nr_samples_ncsn,
                }

                for name, value in log_dict.items():
                    if isinstance(value, jax.Array):
                        value = value.__array__()
                    writer.add_scalar(name, value, nr_samples_ncsn.__array__())
                    if wandb.run is not None:
                        wandb.log({name: value, "global_step": int(nr_samples_ncsn)}, commit=False)
                    rlx_logger.info(
                        f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │",
                        flush=False,
                    )

                rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")

            nr_samples_ncsn = combined_metrics["steps/nr_samples_ncsn"]

            jax.lax.cond(
                nr_samples_ncsn % (50 * config.nr_epochs_ncsn * config.batch_size_ncsn) == 0,
                lambda _: jax.debug.callback(
                    metrics_callback,
                    (combined_metrics, nr_samples_ncsn),
                ),
                lambda _: None,
                operand=None,
            )

            return (energyfn_train_state, (expert_states, expert_actions, expert_next_states, expert_absorbing), rng), None

        num_updates_ncsn = config.total_samples_ncsn // config.nr_epochs_ncsn // config.batch_size_ncsn
        ncsn_learning_iteration_carry_init = (energyfn_train_state, (expert_states, expert_actions, expert_next_states, expert_absorbing), rng)
        ncsn_learning_iteration_carry, _ = jax.lax.scan(ncsn_learning_iteration, 
                                                        ncsn_learning_iteration_carry_init, jnp.arange(num_updates_ncsn))

        energyfn_train_state, (expert_states, expert_actions, expert_next_states, expert_absorbing), rng = ncsn_learning_iteration_carry
        agent_state = cls._agent_state( train_state=None, energyfn_train_state=energyfn_train_state)

        return agent_state


    @classmethod
    def _train_fn(cls, rng, env, writer,
                  agent_conf: NEARAgentConf,
                  agent_state: NEARAgentState = None,
                  mh: MetricsHandler = None):


        config, network, energyfn, tx, energyfn_tx, expert_dataset =\
            (agent_conf.config, agent_conf.network,
             agent_conf.energyfn, agent_conf.tx, agent_conf.energyfn_tx, agent_conf.expert_dataset)

        # extract current agent state
        if agent_state is not None:
            train_state, energyfn_train_state = agent_state.train_state, agent_state.energyfn_train_state
        else:
            train_state, energyfn_train_state = None, None

        assert energyfn_train_state is not None, "A trained energy function must be passed! Something is wrong!"

        if train_state is None:
            rng, _rng1 = jax.random.split(rng, 2)
            init_x = jnp.zeros(env.info.observation_space.shape)
            network_params = network.init(_rng1, init_x)
        else:
            network_params = None

        # init new train states from old params
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params["params"] if train_state is None else train_state.params,
            run_stats=network_params["run_stats"] if train_state is None else train_state.run_stats,
            tx=tx,
        )

        env = cls._wrap_env(env, config)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs)
        obsv, env_state = env.reset(reset_rng)

        train_state_buffer = TrainStateBuffer.create(train_state, config.validation_num)


        #######################################
        """
        RL using energy-based rewards
        """
        #######################################

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            sigmas = jnp.exp(jnp.linspace(jnp.log(config.sigma_begin_ncsn), jnp.log(config.sigma_end_ncsn), config.L_ncsn))

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, energyfn_train_state, sigma_inference_ncsn, last_update_mean_energy, current_lvl_init_energy, env_state, last_obs, train_state_buffer, rng = runner_state

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

                # PREDICT ENERGY BASED REWARD
                if config.state_based:
                    sample = jnp.concatenate([last_obs, obsv], axis=1)
                else:
                    sample = jnp.concatenate([last_obs, action], axis=1)
                
                cond = sigma_inference_ncsn if config.ncsnv1 else sigmas[sigma_inference_ncsn]
                cond = jnp.ones(shape=(sample.shape[0], 1)) * cond
                energy_reward, energy = cls._predict_energy_reward(sample, cond, last_update_mean_energy, energyfn, energyfn_train_state, config.use_running_mean_stand)

                transition = NEARTransition(
                    done, absorbing, action, value, reward, energy_reward, energy, log_prob, last_obs, obsv, info, env_state.additional_carry.traj_state,
                    logged_metrics
                )
                runner_state = (train_state, energyfn_train_state, sigma_inference_ncsn, last_update_mean_energy, current_lvl_init_energy, env_state, obsv, train_state_buffer, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # ANNEAL NOISE LEVEL
            train_state, energyfn_train_state, sigma_inference_ncsn, last_update_mean_energy, current_lvl_init_energy, env_state, last_obs, train_state_buffer, rng = runner_state
            last_update_mean_energy = jnp.mean(traj_batch.energy)
            if config.annealing:
                sigma_inference_ncsn, current_lvl_init_energy = jax.lax.cond(
                    last_update_mean_energy > (current_lvl_init_energy * (1 + config.anneal_threshold)),
                    lambda args: (
                        jnp.clip(args[0] + 1, 0, config.L_ncsn),
                        last_update_mean_energy,
                    ),
                    lambda args: args,
                    (sigma_inference_ncsn, current_lvl_init_energy),
                )

            # CALCULATE ADVANTAGE
            y, _ = network.apply({'params': train_state.params,
                                              'run_stats': train_state.run_stats},
                                             last_obs, mutable=["run_stats"])
            pi, last_val = y

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, absorbing, value, reward, energy_reward, obs, obs_n = (
                        transition.done,
                        transition.absorbing,
                        transition.value,
                        transition.reward,
                        transition.energy_reward,
                        transition.obs,
                        transition.obs_n,
                    )

                    # compute proportion of each reward
                    reward = (config.proportion_env_reward * reward +
                                (1 - config.proportion_env_reward) * energy_reward)

                    delta = reward + config.gamma * next_value * (1 - absorbing) - value
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

            advantages, targets = _calculate_gae(traj_batch, last_val)

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

                train_state, traj_batch, advantages, targets, rng = update_state
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
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )
            train_state = update_state[0]
            rng = update_state[-1]

            counter = ((train_state.step + 1) // config.num_minibatches) // config.update_epochs

            logged_metrics = traj_batch.metrics

            metric = SummaryMetrics(
                mean_episode_return=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 0.0)) / jnp.sum(logged_metrics.done),
                mean_episode_length=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 0.0)) / jnp.sum(logged_metrics.done),
                max_timestep=jnp.max(logged_metrics.timestep * config.num_envs),
            )

            def _evaluation_step():
                def _eval_env(runner_state, unused):
                    train_state, env_state, last_obs, train_state_buffer, rng = runner_state

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

                    runner_state = (train_state, env_state, obsv, train_state_buffer, rng)
                    return runner_state, transition

                rng = runner_state[-1]
                reset_rng = jax.random.split(rng, config.validation_num_envs)
                obsv, env_state = env.reset(reset_rng)
                runner_state_eval = (train_state, env_state, obsv, train_state_buffer, rng)

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

                metric, logged_metrics, loss_info, log_std, last_update_mean_energy, sigma_inference_ncsn = metric
                total_loss, (value_loss, loss_actor, entropy) = loss_info
                log_step = metric.max_timestep.astype('int32')
                log_dict = {
                    "rollout/episode_return": metric.mean_episode_return,
                    "rollout/episode_length": metric.mean_episode_length,
                    "rollout/dones": jnp.mean(logged_metrics.done),
                    "loss/total_loss": jnp.mean(total_loss),
                    "loss/policy_loss": jnp.mean(loss_actor),
                    "loss/critic_loss": jnp.mean(value_loss),
                    "policy/entropy": jnp.mean(entropy),
                    "policy/std": jnp.mean(jnp.exp(log_std)), 
                    "rollout/nr_env_steps": log_step,     
                    "energyfn/mean_energy": last_update_mean_energy,
                    "energyfn/sigma": sigma_inference_ncsn,
                    } 
                for name, value in log_dict.items():
                    writer.add_scalar(name, value.__array__(), log_step.__array__())
                    if wandb.run is not None:
                        wandb.log({name: value.__array__(), "global_step": int(log_step)}, commit=False)
                    rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │", flush=False)

                rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")

            jax.debug.callback(metrics_callback, (metric, logged_metrics, loss_info, train_state.params["log_std"], last_update_mean_energy, sigma_inference_ncsn))

            # DEBUG
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

            runner_state = (train_state, energyfn_train_state, sigma_inference_ncsn, last_update_mean_energy, current_lvl_init_energy, env_state, last_obs, train_state_buffer, rng)
            return runner_state, (metric, validation_metrics)

        rng, _rng = jax.random.split(rng)
        if config.annealing:
            last_update_mean_energy = -1e10
            current_lvl_init_energy = -1e10
        else:
            last_update_mean_energy = 0.0
            current_lvl_init_energy = 0.0
        runner_state = (train_state, energyfn_train_state, config.sigma_inference_ncsn, last_update_mean_energy, current_lvl_init_energy, env_state, obsv, train_state_buffer, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config.num_updates
        )

        agent_state = cls._agent_state(train_state=runner_state[0], energyfn_train_state=runner_state[1])

        return {"agent_state": agent_state,
                "training_metrics": metrics[0],
                "validation_metrics": metrics[1]}


    @classmethod
    def _predict_energy_reward(cls, samples, cond, last_update_mean_energy, energyfn, energyfn_train_state, use_running_mean_stand):
        mutable = ['run_stats'] if use_running_mean_stand else []
        energy, _ = energyfn.apply(
            {'params': energyfn_train_state.params, 'run_stats': energyfn_train_state.run_stats},
            samples, cond,
            mutable=mutable
        )
        energy_reward = 10 * jnp.tanh((energy - last_update_mean_energy)/10)
        return jnp.squeeze(energy_reward), jnp.squeeze(energy)

    @classmethod
    def play_policy(cls, env,
                    agent_conf: NEARAgentConf,
                    agent_state: NEARAgentState,
                    n_envs: int, n_steps=None, render=True,
                    record=False, rng=None, deterministic=False,
                    use_mujoco=False, wrap_env=True,
                    train_state_seed=None, save_traj=False, save_path=None):

        if save_traj:
            assert save_path is not None, "Please provide a save path"
            batch = Batch(
                states=np.zeros((n_envs, n_steps) + env.info.observation_space.shape),
                next_states=np.zeros((n_envs, n_steps) + env.info.observation_space.shape),
                actions=np.zeros((n_envs, n_steps) + env.info.action_space.shape),
                rewards=np.zeros((n_envs, n_steps)),
                terminations=np.zeros((n_envs, n_steps)),
            )

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
                obs_n, reward, absorbing, done, info = env.step(action)
            else:
                obs_n, reward, absorbing, done, info, env_state = env.step(env_state, action)

            # Save trajectory
            if save_traj:
                batch.states[:, i] = obs
                actual_next_state = env_state.additional_carry.final_observation
                batch.next_states[:, i] = jnp.where(done[:, None], actual_next_state, obs_n)
                batch.actions[:, i] = action
                batch.rewards[:, i] = reward
                batch.terminations[:, i] = absorbing
            obs = obs_n


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

        if save_traj:

            def flatten_and_prune(arr):
                flat = arr.reshape(-1, arr.shape[-1])
                nan_mask = ~np.isnan(flat).any(axis=1)
                return flat[nan_mask]

            exp_states = flatten_and_prune(batch.states)
            exp_actions = flatten_and_prune(batch.actions)
            exp_next_states = flatten_and_prune(batch.next_states)            
            exp_absorbing = flatten_and_prune(batch.terminations[:, :, None]).flatten()
            exp_rewards = flatten_and_prune(batch.rewards[:, :, None]).flatten()

            print(f"save path: {save_path}")
            print(f"states shape: {exp_states.shape}")
            print(f"rewards: {exp_rewards.shape}")
            np.savez(f"{save_path}/expert_dataset_{agent_conf.config.env_params.env_name}_{n_envs}_PPO", states=exp_states, actions=exp_actions, 
                next_states=exp_next_states, absorbing=exp_absorbing, rewards=exp_rewards)

    @classmethod
    def play_policy_mujoco(cls, env,
                           agent_conf: NEARAgentConf,
                           agent_state: NEARAgentState,
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