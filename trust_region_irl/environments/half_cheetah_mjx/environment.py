from copy import deepcopy
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from trust_region_irl.environments.half_cheetah_mjx.state import State
from trust_region_irl.environments.half_cheetah_mjx.box_space import BoxSpace
from trust_region_irl.environments.half_cheetah_mjx.viewer import MujocoViewer


class HalfCheetah:
    def __init__(self, render, horizon=1000, feature_fn="base"):
        self.horizon = horizon
        self.feature_fn = feature_fn

        xml_path = (Path(__file__).resolve().parent / "data" / "half_cheetah.xml").as_posix()
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.make_data(self.mjx_model)

        self.nr_intermediate_steps = 1

        self.initial_qpos = jnp.zeros(self.mjx_model.nq)
        self.initial_qvel = jnp.zeros(self.mjx_model.nv)

        action_bounds = self.mj_model.actuator_ctrlrange
        action_low, action_high = action_bounds.T
        self.single_action_space = BoxSpace(low=action_low, high=action_high, shape=(self.mjx_model.nu,), dtype=jnp.float32)
        self.single_observation_space = BoxSpace(
            low=-jnp.inf,
            high=jnp.inf,
            shape=((self.mjx_model.nq - 1) + self.mjx_model.nv,),
            dtype=jnp.float32,
        )
        if self.feature_fn == "state_action":
            feature_dim = self.single_observation_space.shape[0] + self.single_action_space.shape[0]
        else:
            feature_dim = 2
        self.single_features_shape = BoxSpace(low=-jnp.inf, high=jnp.inf, shape=(feature_dim,), dtype=jnp.float32)

        self.forward_reward_weight = 1.0
        self.ctrl_cost_weight: float = 0.1

        self.viewer = None
        if render:
            dt = self.mj_model.opt.timestep * self.nr_intermediate_steps
            self.viewer = MujocoViewer(self.mj_model, dt)
            c_model = deepcopy(self.mj_model)
            c_data = mujoco.MjData(c_model)
            mujoco.mj_step(c_model, c_data, 1)
            self.light_xdir = c_data.light_xdir
            self.light_xpos = c_data.light_xpos
            del c_model, c_data

    def render(self, state):
        env_id = 0
        data = mjx.get_data(self.mj_model, state.data)[env_id]

        data.light_xdir = self.light_xdir
        data.light_xpos = self.light_xpos

        self.viewer.render(data)
        return state

    @partial(jax.vmap, in_axes=(None, 0, None))
    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, eval_mode):
        data = self.mjx_data

        next_observation = jnp.zeros(self.single_observation_space.shape, dtype=jnp.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {
            "rollout/episode_return": reward,
            "rollout/episode_length": 0,
            "env_info/local_vel_x": 0.0,
            "env_info/ctrl_cost": 0.0,
        }
        info_episode_store = {
            "episode_return": reward,
            "episode_length": 0,
        }

        state = State(data, next_observation, next_observation, reward, terminated, truncated, info, info_episode_store, key)
        return self._reset(state)

    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, state):
        data = self.mjx_data
        data = data.replace(qpos=self.initial_qpos, qvel=self.initial_qvel, ctrl=jnp.zeros(self.mjx_model.nu))
        data = mjx.forward(self.mjx_model, data)

        next_observation = self.get_observation(data)
        reward = 0.0
        terminated = False
        truncated = False
        info_episode_store = {
            "episode_return": reward,
            "episode_length": 0,
        }

        return state.replace(
            data=data,
            next_observation=next_observation,
            actual_next_observation=next_observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info_episode_store=info_episode_store,
        )

    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        return self._step(state, action)

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action):
        data, _ = jax.lax.scan(
            f=lambda data, _: (mjx.step(self.mjx_model, data.replace(ctrl=action)), None),
            init=state.data,
            xs=(),
            length=self.nr_intermediate_steps,
        )

        state.info_episode_store["episode_length"] += 1

        next_observation = self.get_observation(data)
        reward, r_info = self.get_reward(data)
        terminated = False
        truncated = state.info_episode_store["episode_length"] >= self.horizon
        done = terminated | truncated

        state.info.update(r_info)
        state.info_episode_store["episode_return"] += reward
        state.info["rollout/episode_return"] = jnp.where(done, state.info_episode_store["episode_return"], state.info["rollout/episode_return"])
        state.info["rollout/episode_length"] = jnp.where(done, state.info_episode_store["episode_length"], state.info["rollout/episode_length"])

        def when_done(_):
            start_state = self._reset(state)
            start_state = start_state.replace(
                actual_next_observation=next_observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )
            return start_state

        def when_not_done(_):
            return state.replace(
                data=data,
                next_observation=next_observation,
                actual_next_observation=next_observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )

        return jax.lax.cond(done, when_done, when_not_done, None)

    def get_observation(self, data):
        position = data.qpos[1:]
        velocity = data.qvel[:]
        observation = jnp.nan_to_num(jnp.concatenate([
            position,
            velocity,
        ]))
        return observation

    def get_reward(self, data):
        local_lin_vel = data.qvel[0]
        forward_reward = self.forward_reward_weight * local_lin_vel
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(data.ctrl))
        reward = jnp.nan_to_num(jnp.clip(forward_reward, max=1e4) - ctrl_cost)

        info = {
            "env_info/local_vel_x": local_lin_vel,
            "env_info/ctrl_cost": ctrl_cost,
        }
        return reward, info

    def close(self):
        if self.viewer:
            self.viewer.close()

    def feature_from_transition(self, observation, action, eps=1e-4):
        def feature_from_transition_base(observation, action, eps=1e-4):
            observation = jnp.asarray(observation, dtype=jnp.float32)
            action = jnp.asarray(action, dtype=jnp.float32)

            squeeze_output = observation.ndim == 1
            if squeeze_output:
                observation = observation[None, :]
                action = action[None, :]

            qpos_without_x_dim = self.mjx_model.nq - 1
            ctrl_cost = jnp.sum(jnp.square(action), axis=-1)
            local_lin_vel = observation[:, qpos_without_x_dim + 0]

            features = jnp.stack([
                local_lin_vel,
                ctrl_cost,
            ], axis=-1)

            if squeeze_output:
                return features[0]
            return features

        def feature_from_transition_stateaction(observation, action, eps=1e-4):
            observation = jnp.asarray(observation, dtype=jnp.float32)
            action = jnp.asarray(action, dtype=jnp.float32)

            squeeze_output = observation.ndim == 1
            if squeeze_output:
                observation = observation[None, :]
                action = action[None, :]

            features = jnp.concatenate([observation, action], axis=-1)

            if squeeze_output:
                return features[0]
            return features

        if self.feature_fn == "base":
            return feature_from_transition_base(observation, action, eps=eps)
        elif self.feature_fn == "state_action":
            return feature_from_transition_stateaction(observation, action, eps=eps)
        else:
            return NotImplementedError
