"""Light-bulb screwing environment (MJX) for the FR3 arm with a Panda gripper.

Task (mirrors the FurnitureBench lamp sub-task): grasp the bulb standing in its
holder, carry it to the socket, insert the screw end into the cavity and seat it
upright. Success = screw tip at the socket seat and bulb upright.

  * Model: scene_mjx_bulb.xml -- FR3 (7 velocity actuators) + Panda hand
    (2 finger position actuators) + free-body bulb + fixed socket + holder.
  * 25-dim observation: bulb_pos_rel_seat(3) + bulb_quat(4) + ee_pos_rel_grasp(3)
    + gripper_width(1) + arm_qpos(7) + arm_qvel(7). Matches the real rig
    (franka_bulbscrew), whose Franka Hand has no fingertip tactile sensing.
  * 5-dim action in [-1,1]: ee velocity (3) + wrist yaw rate (1) + gripper (1).
    Mapped by differential IK to joint velocities (orientation held vertical).
  * No tactile anywhere: the real Franka Hand has no fingertip sensing, so the
    <touch> sensors were removed from the model too and the observation carries
    gripper WIDTH in their place. Both sides must agree exactly on the 25-D
    layout. See SIM_ALIGNMENT.md 2.1.
"""
from copy import deepcopy
from functools import partial
from pathlib import Path

import mujoco
import numpy as np
import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import support as mjx_support

from trust_region_irl.environments.bulbscrew_mjx.state import State
from trust_region_irl.environments.bulbscrew_mjx.box_space import BoxSpace
from trust_region_irl.environments.bulbscrew_mjx.viewer import MujocoViewer

ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
FINGER_JOINT_NAMES = ["finger_joint1", "finger_joint2"]


def _quat_normalize(q, eps=1e-8):
    return q / (jnp.linalg.norm(q) + eps)


def _quat_conj(q):
    return jnp.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_to_rotvec(qe, eps=1e-8):
    qe = _quat_normalize(qe)
    angle = 2.0 * jnp.arccos(jnp.clip(qe[0], -1.0, 1.0))
    s = jnp.sqrt(jnp.maximum(1.0 - qe[0] * qe[0], eps))
    return qe[1:] * (angle / s)


def _quat_error_body(qd, q):
    qd = _quat_normalize(qd)
    q = _quat_normalize(q)
    qe = _quat_mul(qd, _quat_conj(q))
    return _quat_to_rotvec(jnp.where(qe[0] < 0.0, -qe, qe))


class BulbScrew:
    CONTROL_FREQ = 20          # Hz
    # Action scales. These are what the demonstrations are LABELLED with, so the
    # sim and the rig must carry the SAME values or an action of 1.0 means a
    # different speed on each side. Raised 2026-08-31 from the rig's configured
    # 0.10 / 0.75 to the speeds actually ACHIEVED, measured by forward kinematics
    # on the recorded joint encoders: linear p95 0.160 m/s, yaw p95 2.36 rad/s.
    # At the old scales the yaw label saturated on 18.5% of steps (64% of turning
    # steps), discarding how fast the turns really were. See
    # scripts/relabel_real_actions.py; demonstrations relabelled in
    # trirl_dataset/real_expert/relabelled_v1.
    # NOT YET CONFIRMED ACHIEVABLE ON HARDWARE -- servo_ik_node clips joint
    # velocities to 30% of the actuator limits, so before deploying a policy the
    # rig must be commanded at full scale and the achieved rate measured
    # (OPEN_QUESTIONS.md item 7). The rig config still says 0.10 / 0.75.
    MAX_SPEED = 0.17           # m/s ee translation (achieved, p95 0.160)
    MAX_YAW_RATE = 2.6         # rad/s commanded wrist yaw (achieved, p95 2.36)
    GRIP_MAX = 0.04            # m per-finger opening
    KP_ROT = 3.0               # orientation-hold gain (keep gripper vertical)
    KP_NULL = 0.5              # null-space pull toward qhome
    OBS_CLIP = 10.0
    ERR_CLIP = 1.0
    DIVERGE_BOUND = 1.0        # bulb farther than this (m) from the seat -> reset

    # bulb geometry (from the mesh analysis in the model generator)
    TIP_OFF = -0.053           # screw tip, in bulb body frame (z)
    # Grasp point in the bulb body frame (z). The real robot holds the glass
    # body, not the neck capsule. Measured 2026-08-30 with the operator gripping
    # as the task requires: jaws at 58.0 mm on the 60 mm head, TCP 36.7 mm along
    # the bulb axis from the canonical origin. Observation dims 7:10 are the tool
    # relative to THIS point. SIM_CHANGES.md 1.
    NECK_OFF = 0.0367          # grasp point (glass body), in bulb body frame (z)
    SEAT_OFF = jnp.array([0.0, 0.0, 0.018])  # seat point in socket body frame

    # Success is DEPTH held, not a pose test. The old
    # `d_seat + 0.1*upright_err < 0.02` is yaw-invariant (a bulb is a body of
    # revolution), so it cannot tell "resting in the socket mouth" from "screwed
    # tight": on the real rig it fired with the bulb 5.6 mm proud and only 96
    # deg of spin done, with 2 more turns still to go at 2.8 mm/turn. A policy
    # trained against it learns to drop the bulb in the hole and stop.
    # The seat is calibrated at the fully-home pose, so depth is the criterion.
    # SIM_CHANGES.md 2.
    DEPTH_SOLVED = 0.003       # m; screwed down, not merely seated
    HOLD_SECONDS = 1.0         # must stay under DEPTH_SOLVED this long

    # Staging thresholds, so progress can be read per phase instead of inferred
    # from a scalar return. The task is three things in sequence -- get the tip
    # over the hole, get it into the mouth, then turn it down -- and a return
    # curve blurs all three together.
    # Fallen-bulb detection: end the episode instead of running out the horizon
    # on a bulb lying on the table. Thresholds are set from the demonstrations:
    # tilt while gripped peaks at ~37 deg, and the only excursions past 90 deg
    # are single-frame mocap spikes (0.8% of steps), so a 90 deg tilt HELD for
    # half a second is unambiguous. A drop below the plank means it left the
    # table entirely.
    FALL_TILT = 0.7            # upright_err (= 1 - cos tilt); 0.7 is ~72 deg over.
                               # Demos peak at 0.20 (37 deg) while gripped, so this
                               # has ~2x headroom; 1.0 would sit exactly on 90 deg
                               # and a bulb lying flat would not trip it.
    FALL_HOLD_S = 0.5          # must stay tilted this long -- spikes are noise
    FALL_Z = -0.02             # m; bulb body below this has left the table

    XY_TOL = 0.008             # m; tip within this of the socket axis = aligned
    MOUTH_DEPTH = 0.010        # m; tip below this above the seat = in the mouth
                               # (the real bulb rests 5.6 mm proud before screwing)

    def __init__(self, render, horizon=800, reward_style="dense",
                 success_threshold=DEPTH_SOLVED, feature_fn="base"):
        # Sized from the expert episodes truncated at screwed-home: 364 steps at
        # the fastest, 463 median, 619 slowest. 800 is 1.29x the slowest expert
        # and 1.73x the median, so a policy can be ~73% slower than a typical
        # demonstration -- room to fumble and re-approach -- and still finish.
        # Shorter starts cutting off near-successes; longer mostly lets a bad
        # early policy wander before reset. Truncation at the horizon bootstraps
        # (it is not `terminated`), so this adds no bias either way.
        self.horizon = horizon
        self.reward_style = reward_style
        self.success_threshold = success_threshold      # depth, metres
        self.hold_steps = int(round(self.HOLD_SECONDS * self.CONTROL_FREQ))
        self.fall_hold_steps = max(int(round(self.FALL_HOLD_S * self.CONTROL_FREQ)), 1)
        self.feature_fn = feature_fn

        xml_path = (Path(__file__).resolve().parent / "data" / "scene_mjx_bulb.xml").as_posix()
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.make_data(self.mjx_model)

        # 20 Hz control; model timestep 0.01 s -> 5 sim substeps per control step
        self.nr_intermediate_steps = max(
            round((1.0 / self.CONTROL_FREQ) / self.mj_model.opt.timestep), 1)

        # indices
        self.ee_body_id = self.mj_model.body("ee_frame").id
        self.bulb_body_id = self.mj_model.body("bulb").id
        self.socket_body_id = self.mj_model.body("socket").id
        arm_joint_ids = [self.mj_model.joint(n).id for n in ARM_JOINT_NAMES]
        self.arm_qadr = jnp.array(self.mj_model.jnt_qposadr[arm_joint_ids])
        self.arm_dofadr = jnp.array(self.mj_model.jnt_dofadr[arm_joint_ids])
        finger_ids = [self.mj_model.joint(n).id for n in FINGER_JOINT_NAMES]
        self.finger_qadr = jnp.array(self.mj_model.jnt_qposadr[finger_ids])
        self.bulb_qadr = int(self.mj_model.joint("bulb_joint").qposadr[0])
        # velocity-actuator limits (first 7 ctrl entries are the arm)
        self.joint_vel_limits = jnp.array(self.mj_model.actuator_ctrlrange[:7, 1])

        # home arm pose; fingers open; bulb spawn comes from the XML (qpos0)
        # measured real home (franka_bulbscrew ps5_teleop_node.HOME_JOINTS /
        # servo_ik_node nullspace_target); midpoint between bulb and socket
        self.qhome = jnp.array([0.201812, 0.461781, -0.293619,
                                -1.651913, 0.149214, 2.091641, -0.853753])
        initial_qpos = np.array(self.mj_model.qpos0)
        initial_qpos[np.array(self.mj_model.jnt_qposadr[arm_joint_ids])] = np.asarray(self.qhome)
        initial_qpos[np.array(self.mj_model.jnt_qposadr[finger_ids])] = self.GRIP_MAX
        self.initial_qpos = jnp.array(initial_qpos, dtype=jnp.float32)
        self.initial_qvel = jnp.zeros(self.mj_model.nv, dtype=jnp.float32)

        # fixed world anchors (socket is a static body -> read once via CPU forward)
        _d = mujoco.MjData(self.mj_model)
        _d.qpos[:] = np.asarray(initial_qpos)
        mujoco.mj_forward(self.mj_model, _d)
        self.seat_pos = jnp.array(_d.xpos[self.socket_body_id] + np.asarray(self.SEAT_OFF),
                                  dtype=jnp.float32)
        # exactly-vertical grasp orientation: home yaw, tool axis straight down
        R = _d.xmat[self.ee_body_id].reshape(3, 3).copy()
        z_t = np.array([0.0, 0.0, -1.0])
        x_t = R[:, 0] - np.dot(R[:, 0], z_t) * z_t
        x_t /= np.linalg.norm(x_t)
        R_t = np.column_stack([x_t, np.cross(z_t, x_t), z_t])
        q_hold = np.zeros(4)
        mujoco.mju_mat2Quat(q_hold, R_t.ravel())
        self.Q_HOLD = jnp.array(q_hold, dtype=jnp.float32)

        # spaces
        self.single_action_space = BoxSpace(
            low=-jnp.ones(5), high=jnp.ones(5), shape=(5,), dtype=jnp.float32)
        self.single_observation_space = BoxSpace(
            low=-jnp.inf, high=jnp.inf, shape=(25,), dtype=jnp.float32)
        if self.feature_fn == "state_action":
            feature_dim = self.single_observation_space.shape[0] + self.single_action_space.shape[0]
        elif self.feature_fn == "base":
            feature_dim = 5  # [-d_seat, -upright_err, -d_ee_grasp, width, -ctrl]
        elif self.feature_fn == "base_rbf":
            feature_dim = 9  # all of base (incl. -ctrl) + seat_tight, seat_wide,
                             # up_bump, grasp_bump
        elif self.feature_fn == "base_screw":
            feature_dim = 10  # -d_xy, -|depth|, -upright_err, -d_ee_grasp, width,
                              # screw_rate, -ctrl, align_bump, seated_bump, grasp_bump
        else:
            feature_dim = self.single_observation_space.shape[0]
        self.single_features_shape = BoxSpace(
            low=-jnp.inf, high=jnp.inf, shape=(feature_dim,), dtype=jnp.float32)

        self.viewer = None
        if render:
            dt = self.mj_model.opt.timestep * self.nr_intermediate_steps
            self.viewer = MujocoViewer(self.mj_model, dt)
            c_model = deepcopy(self.mj_model)
            c_data = mujoco.MjData(c_model)
            mujoco.mj_step(c_model, c_data, 1)
            self.light_xdir = c_data.light_xdir
            self.light_xpos = c_data.light_xpos

    # ------------------------------------------------------------- dynamics #
    def _differential_ik(self, data, vel_xyz, yaw_rate):
        """Map ee velocity (m/s) + wrist yaw rate to 7 arm joint velocities.

        Analytic body Jacobian + pinv; orientation servoed to the vertical
        grasp quat (so the gripper keeps pointing down) with the commanded yaw
        injected on the z axis. Null-space pull toward qhome; per-joint clamp.
        """
        point = data.xpos[self.ee_body_id]
        jacp, jacr = mjx_support.jac(self.mjx_model, data, point, self.ee_body_id)
        J = jnp.concatenate([jacp.T, jacr.T], axis=0)[:, self.arm_dofadr]  # (6, 7)
        J_pinv = jnp.linalg.pinv(J)

        rot = self.KP_ROT * _quat_error_body(self.Q_HOLD, data.xquat[self.ee_body_id])
        rot = rot.at[2].set(yaw_rate)          # yaw is commanded, not corrected
        twist = jnp.concatenate([vel_xyz, rot])
        dq = J_pinv @ twist

        qnow = data.qpos[self.arm_qadr]
        N = jnp.eye(J.shape[1]) - J_pinv @ J
        dq = dq + N @ (self.KP_NULL * (self.qhome - qnow))
        dq = jnp.nan_to_num(dq, nan=0.0, posinf=0.0, neginf=0.0)
        return jnp.clip(dq, -self.joint_vel_limits, self.joint_vel_limits)

    @partial(jax.vmap, in_axes=(None, 0, None))
    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, eval_mode):
        data = self.mjx_data
        next_observation = jnp.zeros(self.single_observation_space.shape, dtype=jnp.float32)
        info = {
            "rollout/episode_return": 0.0,
            "rollout/episode_length": 0,
            "rollout/is_success": 0.0,
            "rollout/diverged": 0.0,
            "rollout/task_err": 0.0,
            "rollout/reached_socket": 0.0,
            "rollout/spin_turns": 0.0,
            "rollout/depth": 0.0,
            "rollout/fallen": 0.0,
            "env_info/d_seat": 0.0,
            "env_info/d_xy": 0.0,
            "env_info/depth": 0.0,
            "env_info/reached_socket": 0.0,
            "env_info/spin_turns": 0.0,
            "env_info/fallen": 0.0,
            "env_info/upright_err": 0.0,
            "env_info/d_ee_neck": 0.0,
            "env_info/grip_width": 0.0,
            "env_info/task_err": 0.0,
            "env_info/is_success": 0.0,
            "env_info/diverged": 0.0,
        }
        info_episode_store = {"episode_return": 0.0, "episode_length": 0,
                              "steps_at_depth": 0, "prev_yaw": 0.0,
                              "spin_total": 0.0, "reached_socket": 0.0,
                              "steps_tilted": 0}
        state = State(data, next_observation, next_observation, 0.0, False, False,
                      info, info_episode_store, key)
        return self._reset(state)

    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, state):
        data = self.mjx_data
        data = data.replace(qpos=self.initial_qpos, qvel=self.initial_qvel,
                            ctrl=jnp.zeros(self.mjx_model.nu))
        data = mjx.forward(self.mjx_model, data)

        next_observation = self.get_observation(data)
        info = dict(state.info)
        for k in ("env_info/d_seat", "env_info/upright_err", "env_info/d_ee_neck",
                  "env_info/grip_width", "env_info/task_err", "env_info/is_success",
                  "env_info/diverged", "env_info/d_xy", "env_info/depth",
                  "env_info/reached_socket", "env_info/spin_turns",
                  "env_info/fallen"):
            info[k] = 0.0
        return state.replace(
            data=data,
            next_observation=next_observation,
            actual_next_observation=next_observation,
            reward=0.0, terminated=False, truncated=False,
            info=info,
            info_episode_store={"episode_return": 0.0, "episode_length": 0,
                                "steps_at_depth": 0,
                                "prev_yaw": self._stage_terms(data)[2],
                                "spin_total": 0.0, "reached_socket": 0.0,
                                "steps_tilted": 0},
        )

    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        return self._step(state, action)

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action):
        action = jnp.clip(action, -1.0, 1.0)
        vel_xyz = action[:3] * self.MAX_SPEED
        yaw_rate = action[3] * self.MAX_YAW_RATE
        grip = (action[4] + 1.0) * 0.5 * self.GRIP_MAX   # [-1,1] -> [0, 0.04]

        def substep(data, _):
            dq = self._differential_ik(data, vel_xyz, yaw_rate)
            ctrl = jnp.concatenate([dq, jnp.array([grip, grip])])
            data = mjx.step(self.mjx_model, data.replace(ctrl=ctrl))
            return data, None

        data, _ = jax.lax.scan(substep, state.data, xs=(), length=self.nr_intermediate_steps)

        state.info_episode_store["episode_length"] += 1
        next_observation = self.get_observation(data)
        reward, r_info = self.get_reward(data, action)
        # ---- stage tracking: over the hole -> in the mouth -> turned down ----
        d_xy, depth, yaw = self._stage_terms(data)
        engaged = (d_xy < self.XY_TOL) & (depth < self.MOUTH_DEPTH)
        # accumulate |rotation about world z| ONLY while the tip is in the mouth,
        # so carrying the bulb across the table is not counted as screwing
        dyaw = yaw - state.info_episode_store["prev_yaw"]
        dyaw = jnp.arctan2(jnp.sin(dyaw), jnp.cos(dyaw))        # wrap to [-pi, pi]
        state.info_episode_store["spin_total"] += jnp.where(engaged, jnp.abs(dyaw), 0.0)
        state.info_episode_store["prev_yaw"] = yaw
        state.info_episode_store["reached_socket"] = jnp.maximum(
            state.info_episode_store["reached_socket"], engaged.astype(jnp.float32))
        spin_turns = state.info_episode_store["spin_total"] / (2.0 * jnp.pi)
        state.info["env_info/d_xy"] = d_xy
        state.info["env_info/depth"] = depth
        state.info["env_info/reached_socket"] = state.info_episode_store["reached_socket"]
        state.info["env_info/spin_turns"] = spin_turns

        # depth held for HOLD_SECONDS -> screwed home (SIM_CHANGES.md 2)
        at_depth = r_info["env_info/d_seat"] < self.success_threshold
        state.info_episode_store["steps_at_depth"] = jnp.where(
            at_depth, state.info_episode_store["steps_at_depth"] + 1, 0)
        terminated = state.info_episode_store["steps_at_depth"] >= self.hold_steps
        # divergence guard: bulb left a sane region or state went NaN
        bulb_rel = data.xpos[self.bulb_body_id] - self.seat_pos
        diverged = (jnp.nan_to_num(jnp.linalg.norm(bulb_rel), nan=jnp.inf) > self.DIVERGE_BOUND) \
            | jnp.any(jnp.isnan(data.qpos)) | jnp.any(jnp.isnan(data.qvel))
        # Fallen bulb: tipped over and stayed over, or dropped off the table.
        # Routed to `truncated`, NOT `terminated`, on purpose: the dense reward is
        # negative per step, so making a failure END the episode would reward
        # dropping the bulb immediately (the checkpoint-gaming trap from pushT).
        # Truncation bootstraps, so ending early gains the policy nothing -- it
        # only stops the sim wasting a horizon on a bulb lying on the table.
        _, upright_err, _ = self._task_errors(data)
        tilted = (upright_err > self.FALL_TILT) | (data.xpos[self.bulb_body_id][2] < self.FALL_Z)
        state.info_episode_store["steps_tilted"] = jnp.where(
            tilted, state.info_episode_store["steps_tilted"] + 1, 0)
        fallen = state.info_episode_store["steps_tilted"] >= self.fall_hold_steps
        state.info["env_info/fallen"] = fallen.astype(jnp.float32)

        at_horizon = state.info_episode_store["episode_length"] >= self.horizon
        truncated = at_horizon | diverged | fallen
        done = terminated | truncated

        state.info.update(r_info)
        state.info["env_info/diverged"] = diverged.astype(jnp.float32)
        state.info_episode_store["episode_return"] += reward

        # log aggregates only on genuine completions (success or horizon), so
        # divergence-resets can't pollute the means
        # Fallen episodes are EXCLUDED from the return/length means. The dense
        # reward is negative per step, so a short failure scores better than a
        # long success (~-210 vs ~-675): including them would make the return
        # curve RISE as the policy drops the bulb more often. Failure is tracked
        # separately and unconfounded by rollout/fallen and rollout/is_success,
        # both of which are recorded on every episode end.
        log_done = terminated | at_horizon
        state.info["rollout/episode_return"] = jnp.where(log_done, state.info_episode_store["episode_return"], state.info["rollout/episode_return"])
        state.info["rollout/episode_length"] = jnp.where(log_done, state.info_episode_store["episode_length"], state.info["rollout/episode_length"])
        state.info["rollout/is_success"] = jnp.where(done, terminated.astype(jnp.float32), state.info["rollout/is_success"])
        state.info["rollout/diverged"] = jnp.where(done, diverged.astype(jnp.float32), state.info["rollout/diverged"])
        state.info["rollout/task_err"] = jnp.where(done, r_info["env_info/task_err"], state.info["rollout/task_err"])
        # stage aggregates: what fraction of episodes got the bulb into the socket,
        # how far it was turned, and how deep it ended
        state.info["rollout/reached_socket"] = jnp.where(
            done, state.info_episode_store["reached_socket"], state.info["rollout/reached_socket"])
        state.info["rollout/spin_turns"] = jnp.where(done, spin_turns, state.info["rollout/spin_turns"])
        state.info["rollout/depth"] = jnp.where(done, depth, state.info["rollout/depth"])
        state.info["rollout/fallen"] = jnp.where(done, fallen.astype(jnp.float32), state.info["rollout/fallen"])

        def when_done(_):
            start_state = self._reset(state)
            return start_state.replace(
                actual_next_observation=next_observation,
                reward=reward, terminated=terminated, truncated=truncated)

        def when_not_done(_):
            return state.replace(
                data=data,
                next_observation=next_observation,
                actual_next_observation=next_observation,
                reward=reward, terminated=terminated, truncated=truncated)

        return jax.lax.cond(done, when_done, when_not_done, None)

    # --------------------------------------------------------- observation #
    def _bulb_frame(self, data):
        """Bulb pose + derived task points, all from body kinematics."""
        pos = data.xpos[self.bulb_body_id]
        quat = _quat_normalize(data.xquat[self.bulb_body_id])
        w, x, y, z = quat
        # body z axis in world (third column of the rotation matrix)
        axis_z = jnp.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        tip = pos + axis_z * self.TIP_OFF
        neck = pos + axis_z * self.NECK_OFF
        return pos, quat, axis_z, tip, neck

    def _stage_terms(self, data):
        """Lateral offset, insertion depth and bulb yaw -- the three axes the
        task actually progresses along.

        d_xy   : distance from the socket axis (is the tip over the hole?)
        depth  : tip height above the seat; 0 = screwed home, ~0.0056 = resting
        yaw    : bulb rotation about world z, for accumulating screwing progress
        """
        _, quat, _, tip, _ = self._bulb_frame(data)
        rel = tip - self.seat_pos
        d_xy = jnp.linalg.norm(rel[:2])
        depth = rel[2]
        w, x, y, z = quat
        yaw = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return d_xy, depth, yaw

    def _task_errors(self, data):
        _, quat, axis_z, tip, neck = self._bulb_frame(data)
        d_seat = jnp.linalg.norm(tip - self.seat_pos)
        upright_err = 1.0 - axis_z[2]                      # 0 upright .. 2 upside-down
        d_ee_neck = jnp.linalg.norm(data.xpos[self.ee_body_id] - neck)
        clipf = lambda v, hi: jnp.clip(jnp.nan_to_num(v, nan=hi, posinf=hi), 0.0, hi)
        return (clipf(d_seat, self.ERR_CLIP), clipf(upright_err, 2.0),
                clipf(d_ee_neck, self.ERR_CLIP))

    def get_observation(self, data):
        _, quat, _, tip, neck = self._bulb_frame(data)
        bulb_rel_seat = tip - self.seat_pos
        ee_rel_neck = data.xpos[self.ee_body_id] - neck
        width = data.qpos[self.finger_qadr].sum()          # total opening [0, 0.08]
        arm_q = data.qpos[self.arm_qadr]
        arm_v = data.qvel[self.arm_dofadr]
        obs = jnp.concatenate([bulb_rel_seat, quat, ee_rel_neck,
                               jnp.array([width]), arm_q, arm_v])
        obs = jnp.nan_to_num(obs, nan=0.0, posinf=self.OBS_CLIP, neginf=-self.OBS_CLIP)
        return jnp.clip(obs, -self.OBS_CLIP, self.OBS_CLIP)

    def get_reward(self, data, action):
        d_seat, upright_err, d_ee_neck = self._task_errors(data)
        # kept for logging/comparability; NOT the success test any more
        task_err = d_seat + 0.1 * upright_err
        # instantaneous depth test; _step requires it held for HOLD_SECONDS
        is_success = (d_seat < self.success_threshold).astype(jnp.float32)
        width = data.qpos[self.finger_qadr].sum()

        if self.reward_style == "sparse":
            reward = jnp.where(is_success > 0.5, 1.0, 0.0)
        else:
            # dense cost: seat the bulb (dominant), upright, and reach it
            reward = -(10.0 * d_seat + 2.0 * upright_err + 3.0 * d_ee_neck
                       + 0.001 * jnp.sum(jnp.square(action)))
        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=-10.0)

        info = {
            "env_info/d_seat": d_seat,
            "env_info/upright_err": upright_err,
            "env_info/d_ee_neck": d_ee_neck,
            "env_info/grip_width": width,
            "env_info/task_err": task_err,
            "env_info/is_success": is_success,
        }
        return reward, info

    # --------------------------------------------------------------- misc #
    def render(self, state):
        env_id = 0
        data = mjx.get_data(self.mj_model, state.data)[env_id]
        data.light_xdir = self.light_xdir
        data.light_xpos = self.light_xpos
        self.viewer.render(data)
        return state

    def close(self):
        if self.viewer:
            self.viewer.close()

    def feature_from_transition(self, observation, action, eps=1e-4):
        observation = jnp.asarray(observation, dtype=jnp.float32)
        action = jnp.asarray(action, dtype=jnp.float32)
        squeeze_output = observation.ndim == 1
        if squeeze_output:
            observation = observation[None, :]
            action = action[None, :]

        # obs layout (25-D, matches the real rig): bulb_rel_seat 0:3 | quat 3:7 |
        #   ee_rel_grasp 7:10 | width 10 | arm_qpos 11:18 | arm_qvel 18:25
        d_seat = jnp.linalg.norm(observation[:, 0:3], axis=-1)
        qw, qx, qy = observation[:, 3], observation[:, 4], observation[:, 5]
        upright_err = jnp.clip(2.0 * (qx * qx + qy * qy), 0.0, 2.0)  # 1 - R[2,2]
        d_ee_neck = jnp.linalg.norm(observation[:, 7:10], axis=-1)
        width = observation[:, 10]

        if self.feature_fn == "base_rbf":
            # peaked goal-centered bumps: matching the expert requires actually
            # seating the bulb, not matching an average distance (pushT lesson).
            # Strict SUPERSET of `base` -- it carries base's -ctrl term too, so it
            # cannot be worse than base at anything.
            seat_tight = jnp.exp(-(d_seat / 0.02) ** 2)
            seat_wide = jnp.exp(-(d_seat / 0.08) ** 2)
            up_bump = jnp.exp(-(upright_err / 0.10) ** 2)
            grasp_bump = jnp.exp(-(d_ee_neck / 0.05) ** 2)
            ctrl = jnp.sum(jnp.square(action), axis=-1)
            features = jnp.stack([-d_seat, -upright_err, -d_ee_neck, width, -ctrl,
                                  seat_tight, seat_wide, up_bump, grasp_bump], axis=-1)
        elif self.feature_fn == "base_screw":
            # Splits d_seat into the two axes the task actually progresses along
            # and adds the screwing signal, which base/base_rbf cannot express:
            # both are yaw-invariant, so they reward "near the seat" without ever
            # distinguishing RESTING PROUD from TURNED DOWN -- the same blind spot
            # the old success criterion had.
            #
            # NOTE spin here is a RATE, not accumulated turns: features are a
            # function of ONE transition, so total rotation is not available.
            # Screwing is expressed as "turning while engaged", which is: the
            # commanded yaw magnitude gated by being aligned and in the mouth.
            d_xy = jnp.linalg.norm(observation[:, 0:2], axis=-1)
            depth = jnp.abs(observation[:, 2])        # |z| : 0 = screwed home
            # smooth engagement gate, so the gradient survives (a hard AND has none)
            engaged = (jnp.exp(-(d_xy / self.XY_TOL) ** 2)
                       * jnp.exp(-(jnp.maximum(observation[:, 2], 0.0)
                                   / self.MOUTH_DEPTH) ** 2))
            screw_rate = engaged * jnp.abs(action[:, 3])
            align_bump = jnp.exp(-(d_xy / self.XY_TOL) ** 2)
            seated_bump = jnp.exp(-(depth / self.DEPTH_SOLVED) ** 2)
            grasp_bump = jnp.exp(-(d_ee_neck / 0.05) ** 2)
            # upright_err and ctrl are NOT optional. Without upright_err a bulb
            # tilted 45 deg is indistinguishable from an upright one, so the
            # reward would pay a policy for carrying a tilted bulb that cannot be
            # inserted -- and the env's own fall check is built on this very
            # quantity. Without ctrl nothing prefers a controlled motion to
            # full-throttle thrashing.
            ctrl = jnp.sum(jnp.square(action), axis=-1)
            features = jnp.stack([-d_xy, -depth, -upright_err, -d_ee_neck, width,
                                  screw_rate, -ctrl,
                                  align_bump, seated_bump, grasp_bump], axis=-1)
        elif self.feature_fn == "state_action":
            features = jnp.concatenate([observation, action], axis=-1)
        elif self.feature_fn == "base":
            ctrl = jnp.sum(jnp.square(action), axis=-1)
            features = jnp.stack([-d_seat, -upright_err, -d_ee_neck, width, -ctrl], axis=-1)
        else:
            features = observation

        if squeeze_output:
            features = features[0]
        return features
