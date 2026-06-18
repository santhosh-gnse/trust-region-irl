from types import ModuleType
from typing import Any, Dict, Tuple, Union
import numpy as _np
import jax.core as _jax_core
import jax.interpreters.xla as _xla
if not hasattr(_xla, 'pytype_aval_mappings'):
    _xla.pytype_aval_mappings = {}
if _np.ndarray not in _xla.pytype_aval_mappings:
    _xla.pytype_aval_mappings[_np.ndarray] = lambda x: _jax_core.ShapedArray(x.shape, x.dtype)
from loco_mujoco.core.reward import Reward, TargetXVelocityReward
from loco_mujoco.core.reward.default import LocomotionRewardState
from loco_mujoco.core.observations.goals import Goal, GoalRandomRootVelocityState
from loco_mujoco.core.observations.visualizer import RootVelocityArrowVisualizer
from jax._src.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R
import mujoco
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model
import numpy as np
import jax
import jax.numpy as jnp
from loco_mujoco.core.utils import mj_jntname2qposid, mj_jntname2qvelid, mj_jntid2qposid, mj_check_collisions
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast

OVERRIDEVELX, OVERRIDEVELY, OVERRIDEVELYAW = 1.0, 0.0, 0.0

class TargetVelocityGoalRewardOverride(Reward):
    """
    Reward function that computes the reward based on the deviation from the goal velocity. The goal velocity is
    provided as an observation in the environment. The reward is computed as the negative exponential of the squared
    difference between the current velocity and the goal velocity. The reward is computed for the x, y, and yaw
    velocities of the root.

    """

    def __init__(self, env: Any, tracking_w_exp_xy=10.0, tracking_w_exp_yaw=10.0,
                 tracking_w_sum_xy=1.0, tracking_w_sum_yaw=1.0, **kwargs):
        """
        Initialize the reward function.

        Args:
            env (Any): The environment instance.
            tracking_w_exp_xy (float, optional): The exponential weight for xy-tracking reward.
            tracking_w_exp_yaw (float, optional): The exponential weight for yaw-tracking reward.
            **kwargs (Any): Additional keyword arguments.

        """

        super().__init__(env, **kwargs)

        self._free_jnt_name = self._info_props["root_free_joint_xml_name"]
        self._vel_idx = np.array(mj_jntname2qvelid(self._free_jnt_name, env._model))
        self._w_exp_xy = tracking_w_exp_xy
        self._w_exp_yaw = tracking_w_exp_yaw
        self._w_sum_xy = tracking_w_sum_xy
        self._w_sum_yaw = tracking_w_sum_yaw

        # find the goal velocity observation
        assert (("GoalRandomRootVelocity" in env.obs_container) or ("GoalXRootVelocity" in env.obs_container)), \
            f"GoalRandomRootVelocity is the required goal for the reward for{self.__class__.__name__}"

        super().__init__(env, **kwargs)

    def __call__(self,
                 state: Union[np.ndarray, jnp.ndarray],
                 action: Union[np.ndarray, jnp.ndarray],
                 next_state: Union[np.ndarray, jnp.ndarray],
                 absorbing: bool,
                 info: Dict[str, Any],
                 env: Any,
                 model: Union[MjModel, Model],
                 data: Union[MjData, Data],
                 carry: Any,
                 backend: ModuleType) -> Tuple[float, Any]:
        """
        Computes a tracking reward based on the deviation from the goal velocity.Tracking is done on the x, y, and yaw
        velocities of the root.

        Args:
            state (Union[np.ndarray, jnp.ndarray]): Last state.
            action (Union[np.ndarray, jnp.ndarray]): Applied action.
            next_state (Union[np.ndarray, jnp.ndarray]): Current state.
            absorbing (bool): Whether the state is absorbing.
            info (Dict[str, Any]): Additional information.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[float, Any]: The reward for the current transition and the updated carry.
        """
        if backend == np:
            R = np_R
        else:
            R = jnp_R

        goal_state = getattr(carry.observation_states, "GoalXRootVelocity")

        # get root orientation
        root_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self._free_jnt_name)

        assert root_jnt_id != -1, f"Joint {self._free_jnt_name} not found in the model."
        root_jnt_qpos_start_id = model.jnt_qposadr[root_jnt_id]
        root_qpos = backend.squeeze(data.qpos[root_jnt_qpos_start_id:root_jnt_qpos_start_id+7])
        root_quat = R.from_quat(quat_scalarfirst2scalarlast(root_qpos[3:7]))

        # get current local vel of root
        lin_vel_global = backend.squeeze(data.qvel[self._vel_idx])[:3]
        ang_vel_global = backend.squeeze(data.qvel[self._vel_idx])[3:]
        lin_vel_local = root_quat.as_matrix().T @ lin_vel_global
        vel_local = backend.concatenate([lin_vel_local[:2], backend.atleast_1d(ang_vel_global[2])]) # construct vel, x, y and yaw

        # calculate tracking reward
        goal_vel = backend.array([goal_state.goal_vel_x, goal_state.goal_vel_y, goal_state.goal_vel_yaw])
        tracking_reward_xy = backend.exp(-self._w_exp_xy * backend.mean(backend.square(vel_local[:2] - goal_vel[:2])))
        tracking_reward_yaw = backend.exp(-self._w_exp_yaw * backend.mean(backend.square(vel_local[2] - goal_vel[2])))
        total_tracking = self._w_sum_xy * tracking_reward_xy + self._w_sum_yaw * tracking_reward_yaw

        return total_tracking, carry



class LocomotionRewardOverride(TargetVelocityGoalRewardOverride):

    """
    Reward function extending the TargetVelocityGoalReward with typical additional penalties
    and regularization terms for locomotion. This reward is stateful: LocomotionRewardState

    """

    def __init__(self, env: Any, **kwargs):
        """
        Initialize the reward function.

        Args:
            env (Any): The environment instance.
            **kwargs (Any): Additional keyword arguments.

        """
        super().__init__(env, **kwargs)

        model = env._model
        self._free_joint_qpos_ind = np.array(mj_jntname2qposid(self._info_props["root_free_joint_xml_name"], model))
        self._free_joint_qvel_ind = np.array(mj_jntname2qvelid(self._info_props["root_free_joint_xml_name"], model))
        self._free_joint_qpos_mask = np.zeros(model.nq, dtype=bool)
        self._free_joint_qpos_mask[self._free_joint_qpos_ind] = True
        self._free_joint_qvel_mask = np.zeros(model.nv, dtype=bool)
        self._free_joint_qvel_mask[self._free_joint_qvel_ind] = True
        self._foot_names = self._info_props["foot_geom_names"]

        self._floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in self._foot_names]

        # reward coefficients
        self._z_vel_coeff = kwargs.get("z_vel_coeff", 2.0)
        self._roll_pitch_vel_coeff = kwargs.get("roll_pitch_vel_coeff", 5e-2)
        self._roll_pitch_pos_coeff = kwargs.get("roll_pitch_pos_coeff", 2e-1)
        self._nominal_joint_pos_coeff = kwargs.get("nominal_joint_pos_coeff", 0.0)
        self._nominal_joint_pos_names = kwargs.get("nominal_joint_pos_names", None)
        self._joint_position_limit_coeff = kwargs.get("joint_position_limit_coeff", 10.0)
        self._joint_vel_coeff = kwargs.get("joint_vel_coeff", 0.0)
        self._joint_acc_coeff = kwargs.get("joint_acc_coeff", 2e-7)
        self._joint_torque_coeff = kwargs.get("joint_torque_coeff", 2e-5)
        self._action_rate_coeff = kwargs.get("action_rate_coeff", 1e-2)
        self._air_time_max = kwargs.get("air_time_max", 0.0)
        self._air_time_coeff = kwargs.get("air_time_coeff", 0.0)
        self._symmetry_air_coeff = kwargs.get("symmetry_air_coeff", 0.0)
        self._energy_coeff = kwargs.get("energy_coeff", 0.0)

        # get limits and nominal joint positions
        self._limited_joints = np.array(model.jnt_limited, dtype=bool)
        self._limited_joints_qpos_id = model.jnt_qposadr[np.where(self._limited_joints)]
        self._joint_ranges = model.jnt_range[self._limited_joints]
        self._nominal_joint_qpos = env._model.qpos0
        if self._nominal_joint_pos_names is None:
            # take all limited joints
            self._nominal_joint_qpos_id = self._limited_joints_qpos_id
        else:
            self._nominal_joint_qpos_id = np.concatenate([mj_jntname2qposid(name, model)
                                                          for name in self._nominal_joint_pos_names])

    def init_state(self, env: Any,
                   key: Any,
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType):
        """
        Initialize the reward state.

        Args:
            env (Any): The environment instance.
            key (Any): Key for the reward state.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            LocomotionRewardState: The initialized reward state.

        """
        return LocomotionRewardState(last_qvel=data.qvel, last_action=backend.zeros(env.info.action_space.shape[0]),
                                     time_since_last_touchdown=backend.zeros(len(self._foot_ids)))

    def reset(self,
              env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType):
        """
        Reset the reward state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated data and carry.

        """
        reward_state = self.init_state(env, None, model, data, backend)
        carry = carry.replace(reward_state=reward_state)
        return data, carry

    def __call__(self,
                 state: Union[np.ndarray, jnp.ndarray],
                 action: Union[np.ndarray, jnp.ndarray],
                 next_state: Union[np.ndarray, jnp.ndarray],
                 absorbing: bool,
                 info: Dict[str, Any],
                 env: Any,
                 model: Union[MjModel, Model],
                 data: Union[MjData, Data],
                 carry: Any,
                 backend: ModuleType) -> Tuple[float, Any]:
        """
        Based on the tracking reward, this reward function adds typical penalties and regularization terms
        for locomotion.

        Args:
            state (Union[np.ndarray, jnp.ndarray]): Last state.
            action (Union[np.ndarray, jnp.ndarray]): Applied action.
            next_state (Union[np.ndarray, jnp.ndarray]): Current state.
            absorbing (bool): Whether the state is absorbing.
            info (Dict[str, Any]): Additional information.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[float, Any]: The reward for the current transition and the updated carry.
        """

        if backend == np:
            R = np_R
        else:
            R = jnp_R

        # get current reward state
        reward_state = carry.reward_state

        # get global pose quantities
        global_pose_root = data.qpos[self._free_joint_qpos_ind]
        global_pos_root = global_pose_root[:3]
        global_quat_root = global_pose_root[3:]
        global_rot = R.from_quat(quat_scalarfirst2scalarlast(global_quat_root))

        # get global velocity quantities
        global_vel_root = data.qvel[self._free_joint_qvel_ind]

        # get local velocity quantities
        local_vel_root_lin = global_rot.inv().apply(global_vel_root[:3])
        local_vel_root_ang = global_rot.inv().apply(global_vel_root[3:])

        # velocity reward
        if self._z_vel_coeff > 0.0:
            z_vel_reward = self._z_vel_coeff * -(backend.square(local_vel_root_lin[2]))
        else:
            z_vel_reward = 0.0
        if self._roll_pitch_vel_coeff > 0.0:
            roll_pitch_vel_reward = self._roll_pitch_vel_coeff * -backend.square(local_vel_root_ang[:2]).sum()
        else:
            roll_pitch_vel_reward = 0.0

        # position reward
        if self._roll_pitch_pos_coeff > 0.0:
            euler = global_rot.as_euler("xyz")
            roll_pitch_reward = self._roll_pitch_pos_coeff * -backend.square(euler[:2]).sum()
        else:
            roll_pitch_reward = 0.0

        # nominal joint pos reward
        if self._nominal_joint_pos_coeff > 0.0:
            joint_qpos_reward = (self._nominal_joint_pos_coeff *
                                 -backend.square(data.qpos[self._nominal_joint_qpos_id] -
                                                 self._nominal_joint_qpos[self._nominal_joint_qpos_id]).sum())
        else:
            joint_qpos_reward = 0.0

        # joint position limit reward
        if self._joint_position_limit_coeff > 0.0:
            joint_positions = backend.array(data.qpos[self._limited_joints_qpos_id])
            lower_limit_penalty = -backend.minimum(joint_positions - self._joint_ranges[:, 0], 0.0).sum()
            upper_limit_penalty = backend.maximum(joint_positions - self._joint_ranges[:, 1], 0.0).sum()
            joint_position_limit_reward = self._joint_position_limit_coeff * -(lower_limit_penalty + upper_limit_penalty)
        else:
            joint_position_limit_reward = 0.0

        # joint velocity reward
        joint_vel = data.qvel[~self._free_joint_qvel_mask]
        if self._joint_vel_coeff > 0.0:
            joint_vel_reward = self._joint_vel_coeff * -backend.square(joint_vel).sum()
        else:
            joint_vel_reward = 0.0

        # joint acceleration reward
        if self._joint_acc_coeff > 0.0:
            last_joint_vel = reward_state.last_qvel[~self._free_joint_qvel_mask]
            acceleration_norm = backend.sum(backend.square(joint_vel - last_joint_vel) / env.dt)
            acceleration_reward = self._joint_acc_coeff * -acceleration_norm
        else:
            acceleration_reward = 0.0

        # joint torque reward
        if self._joint_torque_coeff > 0.0:
            torque_norm = backend.sum(backend.square(data.qfrc_actuator[~self._free_joint_qvel_mask]))
            torque_reward = self._joint_torque_coeff * -torque_norm
        else:
            torque_reward = 0.0

        # action rate reward
        if self._action_rate_coeff > 0.0:
            action_rate_norm = backend.sum(backend.square(action - reward_state.last_action))
            action_rate_reward = self._action_rate_coeff * -action_rate_norm
        else:
            action_rate_reward = 0.0

        # air time reward
        if self._air_time_coeff > 0.0 or self._symmetry_air_coeff > 0.0:
            air_time_reward = 0.0
            foots_on_ground = backend.zeros(len(self._foot_ids))
            tslt = reward_state.time_since_last_touchdown.copy()
            for i, f_id in enumerate(self._foot_ids):
                foot_on_ground = mj_check_collisions(f_id, self._floor_id, data, backend)
                if backend == np:
                    foots_on_ground[i] = foot_on_ground
                else:
                    foots_on_ground = foots_on_ground.at[i].set(foot_on_ground)

                if backend == np:
                    if foot_on_ground:
                        air_time_reward += (tslt[i] - self._air_time_max)
                        tslt[i] = 0.0
                    else:
                        tslt[i] += env.dt
                else:
                    tslt_i, air_time_reward = jax.lax.cond(foot_on_ground,
                                                           lambda: (0.0, air_time_reward + tslt[i] - self._air_time_max),
                                                           lambda: (tslt[i] + env.dt, air_time_reward))
                    tslt = tslt.at[i].set(tslt_i)

            air_time_reward = self._air_time_coeff * air_time_reward
        else:
            tslt = reward_state.time_since_last_touchdown.copy()
            air_time_reward = 0.0

        # symmetry reward
        if self._symmetry_air_coeff > 0.0:
            symmetry_air_violations = 0.0
            if backend == np:
                if (not foots_on_ground[0] and not foots_on_ground[1]):
                    symmetry_air_violations += 1
                if not foots_on_ground[2] and not foots_on_ground[3]:
                    symmetry_air_violations += 1
            else:
                symmetry_air_violations = jax.lax.cond(jnp.logical_and(jnp.logical_not(foots_on_ground[0]),
                                                                       jnp.logical_not(foots_on_ground[1])),
                                                       lambda: symmetry_air_violations + 1,
                                                       lambda: symmetry_air_violations)

                symmetry_air_violations = jax.lax.cond(jnp.logical_and(jnp.logical_not(foots_on_ground[2]),
                                                                       jnp.logical_not(foots_on_ground[3])),
                                                       lambda: symmetry_air_violations + 1,
                                                       lambda: symmetry_air_violations)

            symmetry_air_reward = self._symmetry_air_coeff * -symmetry_air_violations
        else:
            symmetry_air_reward = 0.0

        # energy reward
        if self._energy_coeff > 0.0:
            energy = backend.sum(backend.abs(joint_vel) * backend.abs(data.qfrc_actuator[~self._free_joint_qvel_mask]))
            energy_reward = self._energy_coeff * -energy
        else:
            energy_reward = 0.0

        # total reward
        tracking_reward, _ = super().__call__(state, action, next_state, absorbing, info,
                                              env, model, data, carry, backend)
        penality_rewards = (z_vel_reward + roll_pitch_vel_reward + roll_pitch_reward + joint_qpos_reward
                            + joint_position_limit_reward + joint_vel_reward + acceleration_reward
                            + torque_reward + action_rate_reward + air_time_reward
                            + symmetry_air_reward + energy_reward)
        
        penalty_rewards = (
            backend.nan_to_num(z_vel_reward)
            + backend.nan_to_num(roll_pitch_vel_reward)
            + backend.nan_to_num(roll_pitch_reward)
            + backend.nan_to_num(joint_qpos_reward)
            + backend.nan_to_num(joint_position_limit_reward)
            + backend.nan_to_num(joint_vel_reward)
            + backend.nan_to_num(acceleration_reward)
            + backend.nan_to_num(torque_reward)
            + backend.nan_to_num(action_rate_reward)
            + backend.nan_to_num(air_time_reward)
            + backend.nan_to_num(symmetry_air_reward)
            + backend.nan_to_num(energy_reward)
        )

        total_reward = backend.nan_to_num(tracking_reward) + penality_rewards
        total_reward = backend.maximum(total_reward, 0.0)

        # update reward state
        reward_state = reward_state.replace(last_qvel=data.qvel, last_action=action, time_since_last_touchdown=tslt)
        carry = carry.replace(reward_state=reward_state)

        return total_reward, carry


class GoalXRootVelocity(Goal, RootVelocityArrowVisualizer):
    """
    A class representing a random root velocity goal.

    This class defines a goal that specifies random velocities for the root body in
    the x, y, and yaw directions.

    Args:
        info_props (Dict): Information properties required for initialization.
        max_x_vel (float): Maximum velocity in the x direction.
        max_y_vel (float): Maximum velocity in the y direction.
        max_yaw_vel (float): Maximum yaw velocity.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self,
                 info_props: Dict,
                 max_x_vel: float = 1.0,
                 max_y_vel: float = 1.0,
                 max_yaw_vel: float = 1.0, **kwargs):

        self._traj_goal_ind = None
        self.max_x_vel = max_x_vel
        self.max_y_vel = max_y_vel
        self.max_yaw_vel = max_yaw_vel
        self.upper_body_xml_name = info_props["upper_body_xml_name"]
        self.free_jnt_name = info_props["root_free_joint_xml_name"]

        # To be initialized from Mujoco
        self._root_body_id = None
        self._root_jnt_qpos_start_id = None

        # call visualizer init
        RootVelocityArrowVisualizer.__init__(self, info_props)

        # call goal init
        n_visual_geoms = self._arrow_n_visual_geoms \
            if "visualize_goal" in kwargs.keys() and kwargs["visualize_goal"] else 0
        super().__init__(info_props, n_visual_geoms=n_visual_geoms, **kwargs)

    def _init_from_mj(self,
                      env: Any,
                      model: Union[MjModel, Model],
                      data: Union[MjData, Data],
                      current_obs_size: int):
        """
        Initialize the goal from Mujoco model and data.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data.
            current_obs_size (int): Current observation size.
        """
        self.min = [-np.inf] * self.dim
        self.max = [np.inf] * self.dim
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + self.dim)])
        self._root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.upper_body_xml_name)
        self._free_jnt_qpos_id = np.array(mj_jntname2qposid(self.free_jnt_name, model))
        self._initialized_from_mj = True

    @property
    def has_visual(self) -> bool:
        """Check if the goal supports visualization."""
        return True

    def init_state(self,
                   env: Any,
                   key: jax.random.PRNGKey,
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType) -> GoalRandomRootVelocityState:
        """
        Initialize the goal state.

        Args:
            env (Any): The environment instance.
            key (jax.random.PRNGKey): Random key for sampling.
            model (Union[MjModel, Any]): The Mujoco model.
            data (Union[MjData, Any]): The Mujoco data.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            GoalRandomRootVelocityState: Initialized state.
        """
        return GoalRandomRootVelocityState(0.0, 0.0, 0.0)

    def reset_state(self,
                    env: Any,
                    model: Union[MjModel, Model],
                    data: Union[MjData, Data],
                    carry: Any,
                    backend: ModuleType) -> Tuple[Union[MjData, Any], Any]:
        """
        Reset the goal state with random velocities.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Any]): The Mujoco model.
            data (Union[MjData, Any]): The Mujoco data.
            carry (Any): Carry object.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            Tuple[Union[MjData, Any], Any]: Updated data and carry.
        """
        key = carry.key
        # if backend == np:
        #     goal_vel = np.random.uniform(
        #         [-self.max_x_vel, -self.max_y_vel, -self.max_yaw_vel],
        #         [self.max_x_vel, self.max_y_vel, self.max_yaw_vel]
        #     )
        # else:
        #     key, subkey = jax.random.split(key)
        #     goal_vel = jax.random.uniform(
        #         subkey,
        #         shape=(3,),
        #         minval=jnp.array([-self.max_x_vel, -self.max_y_vel, -self.max_yaw_vel]),
        #         maxval=jnp.array([self.max_x_vel, self.max_y_vel, self.max_yaw_vel])
        #     )

        goal_vel = [OVERRIDEVELX, OVERRIDEVELY, OVERRIDEVELYAW]

        # goal_state = GoalRandomRootVelocityState(goal_vel[0], goal_vel[1], goal_vel[2])
        goal_state = GoalRandomRootVelocityState(goal_vel[0], 0.0, 0.0)
        observation_states = carry.observation_states.replace(**{self.name: goal_state})
        return data, carry.replace(key=key, observation_states=observation_states)

    def get_obs_and_update_state(self,
                                 env: Any,
                                 model: Union[MjModel, Model],
                                 data: Union[MjData, Data],
                                 carry: Any,
                                 backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Get the current goal observation and update the state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data.
            carry (Any): Carry object.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: Goal observation and updated carry.
        """
        # goal_vel_x = getattr(carry.observation_states, self.name).goal_vel_x
        # goal_vel_y = getattr(carry.observation_states, self.name).goal_vel_y
        # goal_vel_yaw = getattr(carry.observation_states, self.name).goal_vel_yaw
        goal_vel_x = OVERRIDEVELX
        goal_vel_y = OVERRIDEVELY
        goal_vel_yaw = OVERRIDEVELYAW
        goal = backend.array([goal_vel_x, goal_vel_y, goal_vel_yaw])
        goal_visual = backend.array([goal_vel_x, goal_vel_y, 0.0, 0.0, 0.0, goal_vel_yaw])

        if self.visualize_goal:
            carry = self.set_visuals(
                goal_visual, env, model, data, carry, self._root_body_id,
                self._free_jnt_qpos_id, self.visual_geoms_idx, backend
            )

        return goal, carry

    @property
    def dim(self) -> int:
        """Get the dimension of the goal."""
        return 3


class SafeTargetXVelocityReward(TargetXVelocityReward):
    def __call__(self, state, action, next_state, absorbing, info, env,
                 model, data, carry, backend):
        x_vel = backend.squeeze(data.qvel[self._x_vel_idx])
        return backend.nan_to_num(backend.exp(-backend.square(x_vel - self._target_vel))), carry


class SafeLocomotionXVelocityReward(SafeTargetXVelocityReward):

    """
    Reward function extending the TargetXVelocityReward with typical additional penalties
    and regularization terms for locomotion. This reward is stateful: LocomotionRewardState

    """

    def __init__(self, env: Any, **kwargs):
        """
        Initialize the reward function.

        Args:
            env (Any): The environment instance.
            **kwargs (Any): Additional keyword arguments.

        """
        super().__init__(env, **kwargs)

        model = env._model
        self._free_joint_qpos_ind = np.array(mj_jntname2qposid(self._info_props["root_free_joint_xml_name"], model))
        self._free_joint_qvel_ind = np.array(mj_jntname2qvelid(self._info_props["root_free_joint_xml_name"], model))
        self._free_joint_qpos_mask = np.zeros(model.nq, dtype=bool)
        self._free_joint_qpos_mask[self._free_joint_qpos_ind] = True
        self._free_joint_qvel_mask = np.zeros(model.nv, dtype=bool)
        self._free_joint_qvel_mask[self._free_joint_qvel_ind] = True
        self._foot_names = self._info_props["foot_geom_names"]

        self._floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in self._foot_names]

        # reward coefficients
        self._z_vel_coeff = kwargs.get("z_vel_coeff", 2.0)
        self._roll_pitch_vel_coeff = kwargs.get("roll_pitch_vel_coeff", 5e-2)
        self._roll_pitch_pos_coeff = kwargs.get("roll_pitch_pos_coeff", 2e-1)
        self._nominal_joint_pos_coeff = kwargs.get("nominal_joint_pos_coeff", 0.0)
        self._nominal_joint_pos_names = kwargs.get("nominal_joint_pos_names", None)
        self._joint_position_limit_coeff = kwargs.get("joint_position_limit_coeff", 0.0)
        self._joint_vel_coeff = kwargs.get("joint_vel_coeff", 0.0)
        self._joint_acc_coeff = kwargs.get("joint_acc_coeff", 2e-5)
        self._joint_torque_coeff = kwargs.get("joint_torque_coeff", 2e-7)
        self._action_rate_coeff = kwargs.get("action_rate_coeff", 0.1)
        self._air_time_max = kwargs.get("air_time_max", 0.5)
        self._air_time_coeff = kwargs.get("air_time_coeff", 0.1)
        self._symmetry_air_coeff = kwargs.get("symmetry_air_coeff", 0.005)
        self._energy_coeff = kwargs.get("energy_coeff", 5e-5)

        # get limits and nominal joint positions
        self._limited_joints = np.array(model.jnt_limited, dtype=bool)
        self._limited_joints_qpos_id = model.jnt_qposadr[np.where(self._limited_joints)]
        self._joint_ranges = model.jnt_range[self._limited_joints]
        self._nominal_joint_qpos = env._model.qpos0
        if self._nominal_joint_pos_names is None:
            # take all limited joints
            self._nominal_joint_qpos_id = self._limited_joints_qpos_id
        else:
            self._nominal_joint_qpos_id = np.concatenate([mj_jntname2qposid(name, model)
                                                          for name in self._nominal_joint_pos_names])

    def init_state(self, env: Any,
                   key: Any,
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType):
        """
        Initialize the reward state.

        Args:
            env (Any): The environment instance.
            key (Any): Key for the reward state.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            LocomotionRewardState: The initialized reward state.

        """
        return LocomotionRewardState(last_qvel=data.qvel, last_action=backend.zeros(env.info.action_space.shape[0]),
                                     time_since_last_touchdown=backend.zeros(len(self._foot_ids)))

    def reset(self,
              env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType):
        """
        Reset the reward state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated data and carry.

        """
        reward_state = self.init_state(env, None, model, data, backend)
        carry = carry.replace(reward_state=reward_state)
        return data, carry

    def __call__(self,
                 state: Union[np.ndarray, jnp.ndarray],
                 action: Union[np.ndarray, jnp.ndarray],
                 next_state: Union[np.ndarray, jnp.ndarray],
                 absorbing: bool,
                 info: Dict[str, Any],
                 env: Any,
                 model: Union[MjModel, Model],
                 data: Union[MjData, Data],
                 carry: Any,
                 backend: ModuleType) -> Tuple[float, Any]:
        """
        Based on the tracking reward, this reward function adds typical penalties and regularization terms
        for locomotion.

        Args:
            state (Union[np.ndarray, jnp.ndarray]): Last state.
            action (Union[np.ndarray, jnp.ndarray]): Applied action.
            next_state (Union[np.ndarray, jnp.ndarray]): Current state.
            absorbing (bool): Whether the state is absorbing.
            info (Dict[str, Any]): Additional information.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[float, Any]: The reward for the current transition and the updated carry.
        """

        if backend == np:
            R = np_R
        else:
            R = jnp_R

        # get current reward state
        reward_state = carry.reward_state

        # get global pose quantities
        global_pose_root = data.qpos[self._free_joint_qpos_ind]
        global_pos_root = global_pose_root[:3]
        global_quat_root = global_pose_root[3:]
        global_rot = R.from_quat(quat_scalarfirst2scalarlast(global_quat_root))

        # get global velocity quantities
        global_vel_root = data.qvel[self._free_joint_qvel_ind]

        # get local velocity quantities
        local_vel_root_lin = global_rot.inv().apply(global_vel_root[:3])
        local_vel_root_ang = global_rot.inv().apply(global_vel_root[3:])

        # velocity reward
        if self._z_vel_coeff > 0.0:
            z_vel_reward = self._z_vel_coeff * -(backend.square(local_vel_root_lin[2]))
        else:
            z_vel_reward = 0.0
        if self._roll_pitch_vel_coeff > 0.0:
            roll_pitch_vel_reward = self._roll_pitch_vel_coeff * -backend.square(local_vel_root_ang[:2]).sum()
        else:
            roll_pitch_vel_reward = 0.0

        # position reward
        if self._roll_pitch_pos_coeff > 0.0:
            euler = global_rot.as_euler("xyz")
            roll_pitch_reward = self._roll_pitch_pos_coeff * -backend.square(euler[:2]).sum()
        else:
            roll_pitch_reward = 0.0

        # nominal joint pos reward
        if self._nominal_joint_pos_coeff > 0.0:
            joint_qpos_reward = (self._nominal_joint_pos_coeff *
                                 -backend.square(data.qpos[self._nominal_joint_qpos_id] -
                                                 self._nominal_joint_qpos[self._nominal_joint_qpos_id]).sum())
        else:
            joint_qpos_reward = 0.0

        # joint position limit reward
        if self._joint_position_limit_coeff > 0.0:
            joint_positions = backend.array(data.qpos[self._limited_joints_qpos_id])
            lower_limit_penalty = -backend.minimum(joint_positions - self._joint_ranges[:, 0], 0.0).sum()
            upper_limit_penalty = backend.maximum(joint_positions - self._joint_ranges[:, 1], 0.0).sum()
            joint_position_limit_reward = self._joint_position_limit_coeff * -(lower_limit_penalty + upper_limit_penalty)
        else:
            joint_position_limit_reward = 0.0

        # joint velocity reward
        joint_vel = data.qvel[~self._free_joint_qvel_mask]
        if self._joint_vel_coeff > 0.0:
            joint_vel_reward = self._joint_vel_coeff * -backend.square(joint_vel).sum()
        else:
            joint_vel_reward = 0.0

        # joint acceleration reward
        if self._joint_acc_coeff > 0.0:
            last_joint_vel = reward_state.last_qvel[~self._free_joint_qvel_mask]
            acceleration_norm = backend.sum(backend.square(joint_vel - last_joint_vel) / env.dt)
            acceleration_reward = self._joint_acc_coeff * -acceleration_norm
        else:
            acceleration_reward = 0.0

        # joint torque reward
        if self._joint_torque_coeff > 0.0:
            torque_norm = backend.sum(backend.square(data.qfrc_actuator[~self._free_joint_qvel_mask]))
            torque_reward = self._joint_torque_coeff * -torque_norm
        else:
            torque_reward = 0.0

        # action rate reward
        if self._action_rate_coeff > 0.0:
            action_rate_norm = backend.sum(backend.square(action - reward_state.last_action))
            action_rate_reward = self._action_rate_coeff * -action_rate_norm
        else:
            action_rate_reward = 0.0

        # air time reward
        if self._air_time_coeff > 0.0 or self._symmetry_air_coeff > 0.0:
            air_time_reward = 0.0
            foots_on_ground = backend.zeros(len(self._foot_ids))
            tslt = reward_state.time_since_last_touchdown.copy()
            for i, f_id in enumerate(self._foot_ids):
                foot_on_ground = mj_check_collisions(f_id, self._floor_id, data, backend)
                if backend == np:
                    foots_on_ground[i] = foot_on_ground
                else:
                    foots_on_ground = foots_on_ground.at[i].set(foot_on_ground)

                if backend == np:
                    if foot_on_ground:
                        air_time_reward += (tslt[i] - self._air_time_max)
                        tslt[i] = 0.0
                    else:
                        tslt[i] += env.dt
                else:
                    tslt_i, air_time_reward = jax.lax.cond(foot_on_ground,
                                                           lambda: (0.0, air_time_reward + tslt[i] - self._air_time_max),
                                                           lambda: (tslt[i] + env.dt, air_time_reward))
                    tslt = tslt.at[i].set(tslt_i)

            air_time_reward = self._air_time_coeff * air_time_reward
        else:
            tslt = reward_state.time_since_last_touchdown.copy()
            air_time_reward = 0.0

        # symmetry reward
        if self._symmetry_air_coeff > 0.0:
            symmetry_air_violations = 0.0
            if backend == np:
                if (not foots_on_ground[0] and not foots_on_ground[1]):
                    symmetry_air_violations += 1
                if not foots_on_ground[2] and not foots_on_ground[3]:
                    symmetry_air_violations += 1
            else:
                symmetry_air_violations = jax.lax.cond(jnp.logical_and(jnp.logical_not(foots_on_ground[0]),
                                                                       jnp.logical_not(foots_on_ground[1])),
                                                       lambda: symmetry_air_violations + 1,
                                                       lambda: symmetry_air_violations)

                symmetry_air_violations = jax.lax.cond(jnp.logical_and(jnp.logical_not(foots_on_ground[2]),
                                                                       jnp.logical_not(foots_on_ground[3])),
                                                       lambda: symmetry_air_violations + 1,
                                                       lambda: symmetry_air_violations)

            symmetry_air_reward = self._symmetry_air_coeff * -symmetry_air_violations
        else:
            symmetry_air_reward = 0.0

        # energy reward
        if self._energy_coeff > 0.0:
            energy = backend.sum(backend.abs(joint_vel) * backend.abs(data.qfrc_actuator[~self._free_joint_qvel_mask]))
            energy_reward = self._energy_coeff * -energy
        else:
            energy_reward = 0.0

        # total reward
        tracking_reward, _ = super().__call__(state, action, next_state, absorbing, info,
                                              env, model, data, carry, backend)
        penality_rewards = (z_vel_reward + roll_pitch_vel_reward + roll_pitch_reward + joint_qpos_reward
                            + joint_position_limit_reward + joint_vel_reward + acceleration_reward
                            + torque_reward + action_rate_reward + air_time_reward
                            + symmetry_air_reward + energy_reward)

        penality_rewards = sum(
            backend.nan_to_num(r) for r in [
                z_vel_reward,
                roll_pitch_vel_reward,
                roll_pitch_reward,
                joint_qpos_reward,
                joint_position_limit_reward,
                joint_vel_reward,
                acceleration_reward,
                torque_reward,
                action_rate_reward,
                air_time_reward,
                symmetry_air_reward,
                energy_reward
            ]
        )

        total_reward = 0.7*tracking_reward + 0.3*penality_rewards
        # total_reward = backend.maximum(total_reward, 0.0)

        # update reward state
        reward_state = reward_state.replace(last_qvel=data.qvel, last_action=action, time_since_last_touchdown=tslt)
        carry = carry.replace(reward_state=reward_state)

        return total_reward, carry


SafeTargetXVelocityReward.register()
SafeLocomotionXVelocityReward.register()
GoalXRootVelocity.register()
TargetVelocityGoalRewardOverride.register()
LocomotionRewardOverride.register()