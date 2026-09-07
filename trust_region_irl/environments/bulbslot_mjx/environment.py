"""Bulb SLOTTING environment (MJX) for the FR3 arm with a Panda gripper.

Task: the bulb starts ALREADY HELD in the gripper. Carry it to the socket, drop
it into the mouth and release. Nothing is threaded -- this is slotting, not
screwing -- so any final rotation counts and the bulb legitimately rests against
the socket rim at a lean (1.3-6.1 deg across 126 real demonstrations).

  * 25-dim observation, identical layout to the real rig: bulb_pos_rel_seat(3)
    + bulb_quat(4) + ee_pos_rel_grasp(3) + gripper_width(1) + arm_qpos(7)
    + arm_qvel(7). No tactile: the real Franka Hand has none.
  * 5-dim action in [-1,1]: ee velocity (3) + wrist yaw rate (1) + gripper (1).
    Yaw is barely used now that nothing is screwed.
  * Success is POSITION ONLY, d_seat < 20 mm held for 1 s. Uprightness must not
    gate it: a slotted bulb leans by however much it settles, and testing for
    upright rejects a perfectly good slot. Position by contrast repeats to
    0.27 mm across the demonstrations -- the socket sets the stop, not the
    operator.
  * The grasp is a WELD toggled from finger width, not a friction grasp. See
    data/scene_mjx_slot.xml for why and what it costs.
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

from trust_region_irl.environments.bulbslot_mjx.state import State
from trust_region_irl.environments.bulbslot_mjx.box_space import BoxSpace
from trust_region_irl.environments.bulbslot_mjx.viewer import MujocoViewer

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


def _swing_quat(q, eps=1e-8):
    """Strip rotation about the body's own z axis, keeping only the tilt.

    Nothing is threaded in this task, so rotation about the bulb's axis carries
    no information -- but it is not harmless. The real bulb's axis-yaw sits at
    107.6 +- 2.5 deg (set by how the mocap rigid body was defined and how the
    operator places it) while the sim has a single fixed value, a systematic
    30 deg difference on a quantity that says nothing about the task. Left in, a
    discriminator can separate expert from policy on that alone.

    The swing is the minimal rotation taking z to the body's axis, so it is
    yaw-free by construction while preserving the tilt EXACTLY: for a swing,
    qz = 0 and 2(qx^2 + qy^2) = 1 - cos(tilt), which is the same upright_err the
    features and metrics already use.
    """
    w, x, y, z = q
    az = jnp.array([2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)])
    az = az / (jnp.linalg.norm(az) + eps)
    cos_t = jnp.clip(az[2], -1.0, 1.0)
    axis = jnp.array([-az[1], az[0], 0.0])            # z x az
    n = jnp.linalg.norm(axis)
    half = 0.5 * jnp.arccos(cos_t)
    u = jnp.where(n > eps, axis / (n + eps), jnp.array([1.0, 0.0, 0.0]))
    sw = jnp.concatenate([jnp.array([jnp.cos(half)]), jnp.sin(half) * u])
    return _quat_normalize(sw)


def _quat_error_body(qd, q):
    qd = _quat_normalize(qd)
    q = _quat_normalize(q)
    qe = _quat_mul(qd, _quat_conj(q))
    return _quat_to_rotvec(jnp.where(qe[0] < 0.0, -qe, qe))


class BulbSlot:
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
    # MUST match the scales the demonstrations were normalised by, which for the
    # 2026-09-06 slotting session is the rig's configured pair. (The screwing env
    # uses achieved rates; that relabelling is not applied to this dataset.) The
    # linear axes saturate on ~6.5% of steps -- the carry is briefly hand-guided
    # above 0.10 m/s, so transport is labelled slower than performed. The slot
    # itself is unaffected.
    MAX_SPEED = 0.10           # m/s ee translation (rig max_linear_speed)
    MAX_YAW_RATE = 0.75        # rad/s (rig max_yaw_rate); barely used when slotting
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
    DEPTH_SOLVED = 0.020       # m; the bulb sits IN the socket mouth. 20 mm and
                               # not 3 mm: nothing is threaded, so it stops
                               # against the rim -- final depth across 126
                               # demonstrations is 7.2-8.4 mm, spread 0.27 mm.
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

    def __init__(self, render, horizon=400, reward_style="dense",
                 success_threshold=DEPTH_SOLVED, feature_fn="base"):
        # Sized from the demonstrations truncated at SLOTTED: 88 steps at the
        # fastest, 127 median, 186 slowest. 400 is 2.2x the longest, and still
        # 1.4x even if a policy carried 1.58x slower than the operator -- the
        # average excess the 0.10 m/s label cap clips away. Slotting is far
        # shorter than screwing (which needed 800), so a longer horizon would
        # only let a bad policy wander. Truncation bootstraps, so no bias.
        self.horizon = horizon
        self.reward_style = reward_style
        self.success_threshold = success_threshold      # depth, metres
        self.hold_steps = int(round(self.HOLD_SECONDS * self.CONTROL_FREQ))
        self.fall_hold_steps = max(int(round(self.FALL_HOLD_S * self.CONTROL_FREQ)), 1)
        # a released bulb this long away from the seat is not coming back
        self.drop_hold_steps = max(int(round(0.5 * self.CONTROL_FREQ)), 1)
        self.feature_fn = feature_fn

        xml_path = (Path(__file__).resolve().parent / "data" / "scene_mjx_slot.xml").as_posix()
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
        self.grip_weld_id = int(mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, "bulb_grip"))
        # total finger opening below which the jaws count as holding the bulb
        self.GRIP_CLOSED = 0.065
        # geom ids for grasp detection: a real grasp is a TWO-SIDED pinch, i.e.
        # contact between the bulb and BOTH fingertips. Touching with one pad is
        # pushing, not holding.
        gid = lambda n: self.mj_model.geom(n).id
        self.left_pad_ids = jnp.array([gid("left_pad_big"), gid("left_pad_tip")])
        self.right_pad_ids = jnp.array([gid("right_pad_big"), gid("right_pad_tip")])
        self.bulb_geom_ids = jnp.array([gid(n) for n in
            ("bulb_head", "bulb_screw", "bulb_tip_flat", "bulb_neck")])
        # velocity-actuator limits (first 7 ctrl entries are the arm)
        self.joint_vel_limits = jnp.array(self.mj_model.actuator_ctrlrange[:7, 1])

        # home arm pose; fingers open; bulb spawn comes from the XML (qpos0)
        # Measured on the rig 2026-09-06 and driven on the real robot: tool at
        # [0.7369, -0.1828, 0.2096] in fr3_link0, vertical, 81 mm above the
        # bulb's grasp point. Differs from the screwing home -- the socket moved.
        self.qhome = jnp.array([-0.288396, 0.807440, 0.070228,
                                -1.160833, -0.054967, 1.966504, 1.963062])
        initial_qpos = np.array(self.mj_model.qpos0)
        initial_qpos[np.array(self.mj_model.jnt_qposadr[arm_joint_ids])] = np.asarray(self.qhome)
        # fingers CLOSED on the 60 mm head: 55.9 mm measured on the rig. The bulb
        # starts held, so the jaws cannot start open.
        initial_qpos[np.array(self.mj_model.jnt_qposadr[finger_ids])] = 0.02795
        self.initial_qpos = jnp.array(initial_qpos, dtype=jnp.float32)
        self.initial_qvel = jnp.zeros(self.mj_model.nv, dtype=jnp.float32)

        # The real observation's first ten dims are in the SEAT frame, which on
        # this rig is rotated -77.7 deg about z relative to fr3_link0. The sim's
        # socket is axis-aligned, so without rotating here the sim would report
        # those dims in BASE axes -- the norms would agree but the components, and
        # the quaternion's yaw, would not. That hands the discriminator a free way
        # to separate expert from policy on a task-irrelevant convention rather
        # than on behaviour. Measured value from the 2026-09-06 session.
        self.SEAT_QUAT = jnp.array([0.778551, -0.000876, -0.003338, -0.627571])
        _Rm = np.zeros(9)
        mujoco.mju_quat2Mat(_Rm, np.asarray(self.SEAT_QUAT, dtype=np.float64))
        self.R_seat = jnp.array(_Rm.reshape(3, 3), dtype=jnp.float32)   # seat -> base

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
        elif self.feature_fn == "base_slot":
            feature_dim = 8   # -d_xy, -|depth|, -d_ee_grasp, width, -ctrl,
                              # align_bump, seated_bump, grasp_bump
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
            "rollout/depth": 0.0,
            "rollout/fallen": 0.0,
            "rollout/grasped": 0.0,
            "rollout/grasp_frac": 0.0,
            "rollout/grasp_streak": 0.0,
            "rollout/grasp_drops": 0.0,
            "rollout/released": 0.0,
            "rollout/release_d_seat": 0.0,
            "rollout/dropped_early": 0.0,
            "rollout/dropped": 0.0,
            "rollout/min_d_seat": 0.0,
            "rollout/final_d_seat": 0.0,
            "rollout/release_tilt": 0.0,
            "rollout/steps_to_socket": 0.0,
            "env_info/d_seat": 0.0,
            "env_info/d_xy": 0.0,
            "env_info/depth": 0.0,
            "env_info/reached_socket": 0.0,
            "env_info/fallen": 0.0,
            "env_info/grasped": 0.0,
            "env_info/released": 0.0,
            "env_info/dropped": 0.0,
            "env_info/tilt_deg": 0.0,
            "env_info/carry_progress": 0.0,
            "env_info/d_ee_neck": 0.0,
            "env_info/grip_width": 0.0,
            "env_info/is_success": 0.0,
            "env_info/diverged": 0.0,
        }
        info_episode_store = {"episode_return": 0.0, "episode_length": 0,
                              "steps_at_depth": 0, "steps_tilted": 0,
                              "reached_socket": 0.0,
                              "grasped_ever": 0.0, "grasped_steps": 0.0,
                              "grasp_streak_cur": 0.0, "grasp_streak_max": 0.0,
                              "grasp_drops": 0.0, "grasped_prev": 1.0,
                              "released_ever": 0.0, "release_d_seat": 0.0,
                              "release_tilt": 0.0,
                              "min_d_seat": 9.9, "steps_to_socket": 0.0, "steps_loose": 0,
                              "d_seat0": 0.0}
        state = State(data, next_observation, next_observation, 0.0, False, False,
                      info, info_episode_store, key)
        return self._reset(state)

    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, state):
        data = self.mjx_data
        data = data.replace(qpos=self.initial_qpos, qvel=self.initial_qvel,
                            ctrl=jnp.zeros(self.mjx_model.nu))
        # the bulb starts held: activate the weld before the first forward pass
        data = self._grip_weld(data)
        data = mjx.forward(self.mjx_model, data)

        next_observation = self.get_observation(data)
        info = dict(state.info)
        for k in ("env_info/d_seat", "env_info/tilt_deg", "env_info/carry_progress",
                  "env_info/d_ee_neck",
                  "env_info/grip_width", "env_info/is_success",
                  "env_info/diverged", "env_info/d_xy", "env_info/depth",
                  "env_info/reached_socket",
                  "env_info/fallen", "env_info/grasped", "env_info/released",
                  "env_info/dropped"):
            info[k] = 0.0
        return state.replace(
            data=data,
            next_observation=next_observation,
            actual_next_observation=next_observation,
            reward=0.0, terminated=False, truncated=False,
            info=info,
            info_episode_store={"episode_return": 0.0, "episode_length": 0,
                                "steps_at_depth": 0, "steps_tilted": 0,
                                "reached_socket": 0.0,
                                "grasped_ever": 0.0, "grasped_steps": 0.0,
                                "grasp_streak_cur": 0.0, "grasp_streak_max": 0.0,
                                "grasp_drops": 0.0,
                                # 1.0, NOT 0.0: the bulb starts HELD, so a release
                                # on the very first step is a real failure mode and
                                # must be detectable.
                                "grasped_prev": 1.0,
                                "released_ever": 0.0, "release_d_seat": 0.0,
                                "release_tilt": 0.0,
                                "min_d_seat": 9.9, "steps_to_socket": 0.0, "steps_loose": 0,
                                # start distance, so carry_progress can be a
                                # legible 0 -> 1 fraction rather than raw metres
                                "d_seat0": jnp.linalg.norm(
                                    self._bulb_frame(data)[3] - self.seat_pos)},
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
            # re-evaluate the hold every substep from the CURRENT finger width,
            # so opening the jaws releases the bulb without a step of lag
            data = self._grip_weld(data.replace(ctrl=ctrl))
            data = mjx.step(self.mjx_model, data)
            return data, None

        data, _ = jax.lax.scan(substep, state.data, xs=(), length=self.nr_intermediate_steps)

        state.info_episode_store["episode_length"] += 1
        next_observation = self.get_observation(data)
        reward, r_info = self.get_reward(data, action)
        # ---- stage tracking: over the hole -> in the mouth -> turned down ----
        d_xy, depth, yaw = self._stage_terms(data)
        engaged = (d_xy < self.XY_TOL) & (depth < self.MOUTH_DEPTH)

        # grasp: two-sided pinch, from contacts
        grasped = self._grasped(data)
        state.info_episode_store["grasped_ever"] = jnp.maximum(
            state.info_episode_store["grasped_ever"], grasped.astype(jnp.float32))
        state.info_episode_store["grasped_steps"] += grasped.astype(jnp.float32)
        # CONTINUITY, not just total. A latched "ever grasped" counts an accidental
        # brush, and grasp_frac cannot tell one long hold from twenty grabs and
        # drops summing to the same total. The longest unbroken run says whether
        # the gripper can actually carry the bulb; the drop count says how often
        # it lets go after having held it.
        streak = jnp.where(grasped, state.info_episode_store["grasp_streak_cur"] + 1.0, 0.0)
        state.info_episode_store["grasp_streak_cur"] = streak
        state.info_episode_store["grasp_streak_max"] = jnp.maximum(
            state.info_episode_store["grasp_streak_max"], streak)
        # WHERE the bulb was let go is the whole task. A slotting policy that
        # carries the bulb correctly and never opens the jaws has not done it, and
        # one that opens them early has dropped the bulb -- a failure the tilt-based
        # fall check cannot see, because a dropped bulb usually lands upright on
        # the board well under the 72 deg trip and above FALL_Z. Real demonstrations
        # release at d_seat 7.2-8.4 mm; a bad policy releases at hundreds of mm.
        dropped = (state.info_episode_store["grasped_prev"] > 0.5) & (~grasped)
        first_release = dropped & (state.info_episode_store["released_ever"] < 0.5)
        state.info_episode_store["release_d_seat"] = jnp.where(
            first_release, r_info["env_info/d_seat"],
            state.info_episode_store["release_d_seat"])
        state.info_episode_store["released_ever"] = jnp.maximum(
            state.info_episode_store["released_ever"], dropped.astype(jnp.float32))
        state.info["env_info/released"] = state.info_episode_store["released_ever"]
        tilt_deg = r_info["env_info/tilt_deg"]
        state.info_episode_store["release_tilt"] = jnp.where(
            first_release, tilt_deg, state.info_episode_store["release_tilt"])
        # closest the bulb EVER got, which is not the same as where it ended: a
        # policy can reach the socket and then knock the bulb back out
        d_now = r_info["env_info/d_seat"]
        state.info_episode_store["min_d_seat"] = jnp.minimum(
            state.info_episode_store["min_d_seat"], d_now)
        # steps until the bulb first reached the socket mouth (0 = never)
        state.info_episode_store["steps_to_socket"] = jnp.where(
            engaged & (state.info_episode_store["steps_to_socket"] < 0.5),
            state.info_episode_store["episode_length"].astype(jnp.float32),
            state.info_episode_store["steps_to_socket"])
        # legible 0 -> 1 progress along the carry, normalised by the start distance
        state.info["env_info/carry_progress"] = jnp.clip(
            1.0 - d_now / jnp.maximum(state.info_episode_store["d_seat0"], 1e-6), 0.0, 1.0)
        state.info_episode_store["grasp_drops"] += dropped.astype(jnp.float32)
        state.info_episode_store["grasped_prev"] = grasped.astype(jnp.float32)
        state.info["env_info/grasped"] = grasped.astype(jnp.float32)

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

        # DROPPED: the bulb was released nowhere near the socket. This is the
        # characteristic slotting failure and the tilt check above cannot see it
        # -- a dropped bulb usually lands upright on the board, far under the
        # 72 deg trip and above FALL_Z. It is also unrecoverable in practice: the
        # weld only re-engages if the jaws close around the bulb again, which a
        # fumbling policy will not manage. Ending here saves the rest of the
        # horizon, which is the real waste (a fumble at step 20 otherwise runs
        # 380 more steps achieving nothing) -- rather than shortening the horizon
        # for everyone, which would truncate near-successes too.
        loose = (~grasped) & (r_info["env_info/d_seat"] > self.DEPTH_SOLVED)
        state.info_episode_store["steps_loose"] = jnp.where(
            loose, state.info_episode_store["steps_loose"] + 1, 0)
        dropped_bulb = state.info_episode_store["steps_loose"] >= self.drop_hold_steps
        state.info["env_info/dropped"] = dropped_bulb.astype(jnp.float32)
        state.info["env_info/fallen"] = fallen.astype(jnp.float32)

        at_horizon = state.info_episode_store["episode_length"] >= self.horizon
        truncated = at_horizon | diverged | fallen | dropped_bulb
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
        state.info["rollout/task_err"] = jnp.where(done, r_info["env_info/d_seat"], state.info["rollout/task_err"])
        # stage aggregates: what fraction of episodes got the bulb into the socket,
        # how far it was turned, and how deep it ended
        state.info["rollout/reached_socket"] = jnp.where(
            done, state.info_episode_store["reached_socket"], state.info["rollout/reached_socket"])
        state.info["rollout/depth"] = jnp.where(done, depth, state.info["rollout/depth"])
        state.info["rollout/fallen"] = jnp.where(done, fallen.astype(jnp.float32), state.info["rollout/fallen"])
        # did the gripper ever actually hold the bulb, and for what fraction of
        # the episode -- a policy that grabs and drops repeatedly looks very
        # different from one that keeps hold
        state.info["rollout/grasped"] = jnp.where(
            done, state.info_episode_store["grasped_ever"], state.info["rollout/grasped"])
        state.info["rollout/grasp_frac"] = jnp.where(
            done, state.info_episode_store["grasped_steps"]
                  / jnp.maximum(state.info_episode_store["episode_length"], 1),
            state.info["rollout/grasp_frac"])
        state.info["rollout/grasp_streak"] = jnp.where(
            done, state.info_episode_store["grasp_streak_max"], state.info["rollout/grasp_streak"])
        state.info["rollout/grasp_drops"] = jnp.where(
            done, state.info_episode_store["grasp_drops"], state.info["rollout/grasp_drops"])
        rel = state.info_episode_store["released_ever"]
        rds = state.info_episode_store["release_d_seat"]
        state.info["rollout/released"] = jnp.where(done, rel, state.info["rollout/released"])
        state.info["rollout/release_d_seat"] = jnp.where(done, rds, state.info["rollout/release_d_seat"])
        # released the bulb while it was nowhere near the socket = dropped it
        state.info["rollout/dropped_early"] = jnp.where(
            done, ((rel > 0.5) & (rds > self.DEPTH_SOLVED)).astype(jnp.float32),
            state.info["rollout/dropped_early"])
        st_ = state.info_episode_store
        # how often the policy fumbles the bulb, recorded before the auto-reset
        # clobbers the per-step flag
        state.info["rollout/dropped"] = jnp.where(
            done, dropped_bulb.astype(jnp.float32), state.info["rollout/dropped"])
        state.info["rollout/min_d_seat"] = jnp.where(done, st_["min_d_seat"], state.info["rollout/min_d_seat"])
        state.info["rollout/final_d_seat"] = jnp.where(done, d_now, state.info["rollout/final_d_seat"])
        state.info["rollout/release_tilt"] = jnp.where(done, st_["release_tilt"], state.info["rollout/release_tilt"])
        state.info["rollout/steps_to_socket"] = jnp.where(done, st_["steps_to_socket"], state.info["rollout/steps_to_socket"])

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
        rel = self.R_seat.T @ (tip - self.seat_pos)   # seat frame, as the obs is
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

    def _grip_weld(self, data):
        """Hold the bulb only while the jaws are closed on it.

        The weld stands in for a friction grasp, which MJX cannot do reliably on
        a smooth sphere. Toggling it from the finger width keeps the release
        meaningful: an always-on weld could never let go, and every real
        demonstration ends with exactly one open transition.
        """
        held = data.qpos[self.finger_qadr].sum() < self.GRIP_CLOSED
        # jnp.asarray: at reset, mjx_data.eq_active is still a numpy array and
        # has no .at[]; inside the jitted step it is already a JAX array.
        eq_active = jnp.asarray(data.eq_active)
        return data.replace(eq_active=eq_active.at[self.grip_weld_id]
                            .set(held.astype(eq_active.dtype)))

    def _grasped(self, data):
        """True while the bulb is held.

        The screwing env reads this from a two-sided pad contact, but here the
        grasp is a weld and produces no contacts at all, so the same test would
        always report False. The weld's own activation is the ground truth, and
        it is a function of finger width -- which is exactly what closing on a
        60 mm head produces and closing on air does not.
        """
        return data.qpos[self.finger_qadr].sum() < self.GRIP_CLOSED

    def get_observation(self, data):
        _, quat, _, tip, neck = self._bulb_frame(data)
        # into the SEAT frame, matching the real rig's convention
        bulb_rel_seat = self.R_seat.T @ (tip - self.seat_pos)
        ee_rel_neck = self.R_seat.T @ (data.xpos[self.ee_body_id] - neck)
        quat = _swing_quat(_quat_mul(_quat_conj(self.SEAT_QUAT), quat))
        width = data.qpos[self.finger_qadr].sum()          # total opening [0, 0.08]
        arm_q = data.qpos[self.arm_qadr]
        arm_v = data.qvel[self.arm_dofadr]
        obs = jnp.concatenate([bulb_rel_seat, quat, ee_rel_neck,
                               jnp.array([width]), arm_q, arm_v])
        obs = jnp.nan_to_num(obs, nan=0.0, posinf=self.OBS_CLIP, neginf=-self.OBS_CLIP)
        return jnp.clip(obs, -self.OBS_CLIP, self.OBS_CLIP)

    def get_reward(self, data, action):
        d_seat, upright_err, d_ee_neck = self._task_errors(data)
        # SLOTTING: position only. A slotted bulb rests against the socket rim at
        # whatever angle it settles (1.3-6.1 deg measured), so an uprightness term
        # would reject a perfectly good slot in the success test, and in the cost
        # it would teach the policy to fight a lean that physics imposes.
        # upright_err is still computed and logged, just not scored.
        task_err = d_seat
        is_success = (d_seat < self.success_threshold).astype(jnp.float32)
        width = data.qpos[self.finger_qadr].sum()

        if self.reward_style == "sparse":
            reward = jnp.where(is_success > 0.5, 1.0, 0.0)
        else:
            # Dense cost for EVALUATION only (env_reward_frac = 0). Two terms
            # from the screwing version are deliberately gone:
            #   upright_err -- a slotted bulb legitimately leans (1.3-6.1 deg);
            #   d_ee_neck   -- the weld pins it to ~0 for the whole carry, so it
            #     carries no gradient when it would matter, and AFTER release it
            #     grows as the tool retreats. The operator retreats 166 mm on
            #     average, so a +3.0 coefficient penalises the correct ending.
            # (The FEATURE sets may still include d_ee_neck: IRL learns its sign,
            # whereas this hand-written coefficient fixes it wrongly.)
            reward = -(10.0 * d_seat + 0.001 * jnp.sum(jnp.square(action)))
        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=-10.0)

        info = {
            "env_info/d_seat": d_seat,
            # tilt in DEGREES: directly comparable to the 1.3-6.1 deg the real
            # demonstrations settle at. Logged, never scored.
            "env_info/tilt_deg": jnp.degrees(
                jnp.arccos(jnp.clip(1.0 - upright_err, -1.0, 1.0))),
            "env_info/d_ee_neck": d_ee_neck,
            "env_info/grip_width": width,
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
            # This is a strict SUPERSET of `base` -- it carries base's -ctrl term
            # too. Dropping it (as the pushT version does) left the set unable to
            # tell a gentle motion from full-throttle thrashing, which is a
            # capability `base` had.
            seat_tight = jnp.exp(-(d_seat / 0.02) ** 2)
            seat_wide = jnp.exp(-(d_seat / 0.08) ** 2)
            up_bump = jnp.exp(-(upright_err / 0.10) ** 2)
            grasp_bump = jnp.exp(-(d_ee_neck / 0.05) ** 2)
            ctrl = jnp.sum(jnp.square(action), axis=-1)
            features = jnp.stack([-d_seat, -upright_err, -d_ee_neck, width, -ctrl,
                                  seat_tight, seat_wide, up_bump, grasp_bump], axis=-1)
        elif self.feature_fn == "base_slot":
            # The slotting counterpart of base_screw. Two deliberate omissions:
            #   upright_err -- a slotted bulb rests against the rim at whatever
            #     angle it settles (1.3-6.1 deg measured), so scoring uprightness
            #     penalises correct behaviour;
            #   screw_rate  -- nothing is threaded, so turning while engaged is
            #     not part of the task.
            # What remains is what the task actually is: get the tip over the
            # hole, get it down into the mouth, and hold on until then.
            d_xy = jnp.linalg.norm(observation[:, 0:2], axis=-1)
            depth = jnp.abs(observation[:, 2])
            ctrl = jnp.sum(jnp.square(action), axis=-1)
            align_bump = jnp.exp(-(d_xy / self.XY_TOL) ** 2)
            seated_bump = jnp.exp(-(depth / self.DEPTH_SOLVED) ** 2)
            grasp_bump = jnp.exp(-(d_ee_neck / 0.05) ** 2)
            features = jnp.stack([-d_xy, -depth, -d_ee_neck, width, -ctrl,
                                  align_bump, seated_bump, grasp_bump], axis=-1)
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
            # upright_err and ctrl are NOT optional here. Without upright_err a
            # bulb tilted 45 deg is indistinguishable from an upright one, so the
            # reward would pay a policy for carrying a tilted bulb that cannot be
            # inserted -- and the env's own fall check is built on this very
            # quantity. Without ctrl there is nothing preferring a controlled
            # motion to full-throttle thrashing.
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
