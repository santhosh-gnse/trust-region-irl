"""Pizza-sauce spreading environment (MJX) for the FR3 with the pusher tool.

Task: sweep the tool over the pizza so the sauce is spread across the whole
surface. The demonstrations are outward spirals from the centre to the rim.

  * Model: scene_mjx_pizza.xml -- FR3 (7 velocity actuators) + the custom pusher
    (a 20x20x125 mm rod with a 15 mm sphere tip, TCP at the sphere centre), and
    a pizza board. Same arm and tool as pusht_mjx.
  * 2-D action in [-1,1] -> x MAX_SPEED -> differential IK, exactly as pushT:
    height and orientation are held by the IK, so the policy commands only the
    planar velocity.
  * Observation matches the real dataset EXACTLY:
        ee_pos_rel_home(3) + arm_qpos(7) + arm_qvel(7) = 17
    `home` is the board centre, so obs[0:2] is the tool position relative to the
    pizza centre.

COVERAGE (obs_mode="coverage", 18-D). The 17-D observation carries no sauce, so a
reward over it cannot express "the pizza got covered" -- coverage is a property
of where the tool has BEEN, and features see one transition at a time. Appending
a coverage channel puts that history INTO the state, which is what lets a
per-transition feature see it. It is computable offline from the logged tool path
for the real data too, so both sides can carry it.

It is a MODEL of coverage (tool-swept area), not measured sauce: real sauce
smears wider than the tool. And a deployed policy needs the same accumulator
running in its observation node.
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

from trust_region_irl.environments.pizza_mjx.state import State
from trust_region_irl.environments.pizza_mjx.box_space import BoxSpace
from trust_region_irl.environments.pizza_mjx.viewer import MujocoViewer

ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]

# measured on the rig (config/pizza_robot.yaml)
QHOME = np.array([-0.423000, 0.625100, 0.423600, -1.912700, -0.390000, 2.456700, -1.858600])
HOME_POS = np.array([0.6247, -0.0094, 0.0641])          # board centre x/y, 20 mm above it
GOAL_QUAT_EE = np.array([0.0, 0.7071068, 0.7071068, 0.0])   # tool-down, same as pushT


def _quat_normalize(q, eps=1e-8):
    return q / (jnp.linalg.norm(q) + eps)


def _quat_conj(q):
    return jnp.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                      w1*x2 + x1*w2 + y1*z2 - z1*y2,
                      w1*y2 - x1*z2 + y1*w2 + z1*x2,
                      w1*z2 + x1*y2 - y1*x2 + z1*w2])


def _quat_to_rotvec(qe, eps=1e-8):
    qe = _quat_normalize(qe)
    angle = 2.0 * jnp.arccos(jnp.clip(qe[0], -1.0, 1.0))
    s = jnp.sqrt(jnp.maximum(1.0 - qe[0] * qe[0], eps))
    return qe[1:] * (angle / s)


def _quat_error_body(qd, q):
    qe = _quat_mul(_quat_normalize(qd), _quat_conj(_quat_normalize(q)))
    return _quat_to_rotvec(jnp.where(qe[0] < 0.0, -qe, qe))


class PizzaSpread:
    CONTROL_FREQ = 20          # Hz, the rig's rate
    MAX_SPEED = 0.354         # = retiming v_peak 0.30 x fastest tempo 1.18; the
                          # demonstrations really do run at ~235 mm/s, so the old
                          # 100 mm/s scale clipped 76% of the expert actions
    KP_ORI = 0.5               # null-space pull toward qhome
    KP_Z = 5.0                 # height-hold gain. pushT leaves this at an implicit
                               # 1.0, which drifts ~10 mm at full planar speed; the
                               # task is meant to be planar, so hold it properly.
    OBS_CLIP = 10.0

    # Pizza geometry: centre and radius are REGRESSED from the demonstrations.
    # The demonstrated sweep is not centred on the tool's home pose -- it sits
    # 15 mm behind it in x. Fitting the disc to the sweep instead of assuming it
    # is centred on home lets the same demonstrations cover more of the pizza
    # (80.2% vs 75.7%) with a smaller pizza (146 mm vs 156 mm), while still
    # containing 99.7% of every logged sample. Fitted in the FK frame, because
    # that is the frame the observation is built in.
    PIZZA_OFF = np.array([-0.010, 0.000])    # pizza centre, relative to tool home
    PIZZA_R = 0.146
    TOOL_R = 0.015             # tool sphere radius; the swept footprint
    Z_HOLD = HOME_POS[2]       # IK holds the tool at the home height

    # Early termination for wandering off the pizza. The margin is set strictly
    # above what any real demonstration ever reaches -- max radial excess beyond
    # the fitted edge is 14 mm over all 117 episodes -- so a real trajectory can
    # NEVER cross it, regardless of hold time. The hold duration is calibrated
    # against the arm's own recovery dynamics: driving the tool constantly outward
    # does not diverge monotonically -- the null-space qhome pull drags it back
    # after a few dozen steps -- so a policy that is actually stuck off-pizza (not
    # just transiently overshooting) sustains beyond this margin for 30+ steps in
    # practice, well clear of the 10-step hold.
    WANDER_MARGIN = 0.025      # m beyond PIZZA_R (real-data max excess is 14 mm)
    WANDER_HOLD_S = 0.5        # s continuously beyond that margin before truncating

    # Imitation-quality logging metric, from the TRIRL paper's per-step score
    # exp(-(v_pi - v_piE)^2), adapted to joint POSITIONS here (as suggested) since
    # this task has no natural single "upper-body velocity" scalar. It needs a
    # per-step expert reference to compare against, which this task doesn't have
    # for free -- there is no single demonstrated clip, and episodes vary in
    # length (179-396 steps) -- so REF_QPOS is a reference trajectory pooled from
    # all 117 tempo=1.0 demonstrations, binned by fractional PROGRESS through the
    # episode (100 bins) rather than raw step index, which is what makes pooling
    # variable-length episodes meaningful at all. Sanity check: scoring each real
    # demo against this pooled reference gives 0.98 +- 0.008 (95% range 0.95-0.99)
    # -- a real ceiling for "perfect imitation" that is comfortably below 1.0
    # because it is pooled/averaged, not because of any noise in the metric itself.
    # EXPECTED_LENGTH (median real episode length) turns the policy's own elapsed
    # step count into that same progress fraction, so a policy that is much
    # faster or slower than the expert's typical pace gets compared against the
    # WRONG point in the reference and is correctly penalised for it.
    EXPECTED_LENGTH = 256       # steps; median real (tempo=1.0) episode length
    REF_N_BINS = 100

    # coverage grid -- LOGGING ONLY, never a reward feature (see get_features)
    N_RAD, N_ANG = 8, 24       # 192 cells; ~18 mm radial x ~38 mm arc at the rim,
                               # comparable to the 30 mm tool footprint

    def __init__(self, render, horizon=500, reward_style="dense",
                 success_threshold=0.90, feature_fn="base", obs_mode="plain", feature_encoder=""):
        self.horizon = horizon
        self.reward_style = reward_style
        self.success_threshold = success_threshold      # coverage fraction
        self.feature_fn = feature_fn
        # feature_fn="learned": a frozen self-supervised encoder of the raw 17-D state (no coverage)
        self.learned_features = None
        if feature_fn == "learned":
            from trust_region_irl.representation.encoder import FrozenFeatures
            if not feature_encoder:
                raise ValueError("feature_fn='learned' needs environment.feature_encoder=<encoder.pkl>")
            self.learned_features = FrozenFeatures(feature_encoder)
        self.obs_mode = obs_mode                        # "plain" (17) | "coverage" (18)

        xml = (Path(__file__).resolve().parent / "data" / "scene_mjx_pizza.xml").as_posix()
        self.mj_model = mujoco.MjModel.from_xml_path(xml)
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.make_data(self.mjx_model)

        self.nr_intermediate_steps = max(
            round((1.0 / self.CONTROL_FREQ) / self.mj_model.opt.timestep), 1)

        self.ee_body_id = self.mj_model.body("ee_frame").id
        arm_ids = [self.mj_model.joint(n).id for n in ARM_JOINT_NAMES]
        self.arm_qadr = jnp.array(self.mj_model.jnt_qposadr[arm_ids])
        self.arm_dofadr = jnp.array(self.mj_model.jnt_dofadr[arm_ids])
        self.joint_vel_limits = jnp.array(self.mj_model.actuator_ctrlrange[:7, 1])

        self.qhome = jnp.array(QHOME, dtype=jnp.float32)
        self.GOAL_QUAT_EE = jnp.array(GOAL_QUAT_EE, dtype=jnp.float32)
        initial_qpos = np.array(self.mj_model.qpos0)
        initial_qpos[np.array(self.mj_model.jnt_qposadr[arm_ids])] = QHOME
        self.initial_qpos = jnp.array(initial_qpos, dtype=jnp.float32)
        self.initial_qvel = jnp.zeros(self.mj_model.nv, dtype=jnp.float32)

        # home tool position, read from the model rather than trusting the constant
        _d = mujoco.MjData(self.mj_model)
        _d.qpos[:] = np.asarray(initial_qpos)
        mujoco.mj_forward(self.mj_model, _d)
        self.home_pos = jnp.array(_d.xpos[self.ee_body_id], dtype=jnp.float32)
        self.pizza_off = jnp.array(self.PIZZA_OFF, dtype=jnp.float32)
        self.cell_centres = jnp.array(self._cell_centres(), dtype=jnp.float32)
        ref_path = Path(__file__).resolve().parent / "data" / "expert_joint_reference.npy"
        self.ref_qpos = jnp.array(np.load(ref_path), dtype=jnp.float32)   # (REF_N_BINS, 7)

        obs_dim = 17 + (1 if obs_mode == "coverage" else 0)
        self.single_action_space = BoxSpace(
            low=-jnp.ones(2), high=jnp.ones(2), shape=(2,), dtype=jnp.float32)
        self.single_observation_space = BoxSpace(
            low=-jnp.inf, high=jnp.inf, shape=(obs_dim,), dtype=jnp.float32)
        if feature_fn == "state_action":
            fd = obs_dim + 2
        elif feature_fn == "base":
            fd = 3
        elif feature_fn == "base_rbf":
            fd = 6
        elif feature_fn == "spread":
            fd = 8
        elif feature_fn == "learned":
            fd = self.learned_features.d   # frozen SSL encoder output; reward stays linear in it
        else:
            fd = obs_dim
        self.single_features_shape = BoxSpace(
            low=-jnp.inf, high=jnp.inf, shape=(fd,), dtype=jnp.float32)

        self.viewer = MujocoViewer(self.mj_model, 1 / self.CONTROL_FREQ) if render else None
        if self.viewer is not None:
            self.light_xdir = deepcopy(self.mj_data.light_xdir)
            self.light_xpos = deepcopy(self.mj_data.light_xpos)

    # --------------------------------------------------------- coverage #
    # Coverage is a LOGGING metric. It is a model of the swept area, not measured
    # sauce -- nothing on the real rig reports whether the pizza actually got
    # covered -- so no reward feature is allowed to read it.
    def _cell_centres(self):
        """Centres of the polar cells, in pizza-centred coordinates."""
        ir = (np.arange(self.N_RAD) + 0.5) / self.N_RAD * self.PIZZA_R
        ia = (np.arange(self.N_ANG) + 0.5) / self.N_ANG * 2 * np.pi - np.pi
        return np.stack([np.outer(ir, np.cos(ia)),
                         np.outer(ir, np.sin(ia))], -1).reshape(-1, 2)

    def _cell_of(self, xy):
        """Polar cell index of a planar position relative to the pizza centre."""
        q = xy - self.pizza_off
        r = jnp.linalg.norm(q)
        th = jnp.arctan2(q[1], q[0])
        ir = jnp.clip((r / self.PIZZA_R * self.N_RAD).astype(jnp.int32), 0, self.N_RAD - 1)
        ia = jnp.clip(((th + jnp.pi) / (2 * jnp.pi) * self.N_ANG).astype(jnp.int32),
                      0, self.N_ANG - 1)
        on_pizza = r <= self.PIZZA_R
        return ir * self.N_ANG + ia, on_pizza

    def _mark(self, visited, xy):
        """Mark every cell the tool DISC touches, not just the one under its centre.

        The tool is 30 mm across and a rim cell is ~38 mm wide, so crediting only
        the centre cell throws away most of what a pass actually wipes: on the real
        demonstrations that reads 50% coverage where the footprint model reads 79%."""
        idx, on = self._cell_of(xy)
        visited = jnp.where(on, visited.at[idx].set(1.0), visited)
        d = jnp.linalg.norm(self.cell_centres - (xy - self.pizza_off), axis=-1)
        return jnp.maximum(visited, (d <= self.TOOL_R).astype(jnp.float32))

    # --------------------------------------------------------- kinematics #
    def _differential_ik(self, data, vel_xy):
        """Planar tool velocity -> arm joint velocities. Port of pushT's, which is
        in turn the generator's differential_IK: analytic Jacobian, damped pinv,
        height and orientation held by servo terms, null-space pull to qhome."""
        point = data.xpos[self.ee_body_id]
        jacp, jacr = mjx_support.jac(self.mjx_model, data, point, self.ee_body_id)
        J = jnp.concatenate([jacp.T, jacr.T], axis=0)[:, self.arm_dofadr]
        J_pinv = jnp.linalg.pinv(J)
        ee_pos = data.xpos[self.ee_body_id]
        goal_vec = _quat_error_body(self.GOAL_QUAT_EE, data.xquat[self.ee_body_id])
        twist = jnp.concatenate([vel_xy,
                                 jnp.array([self.KP_Z * (self.Z_HOLD - ee_pos[2])]),
                                 goal_vec])
        dq = J_pinv @ twist
        N = jnp.eye(J.shape[1]) - J_pinv @ J
        dq = dq + N @ (self.KP_ORI * (self.qhome - data.qpos[self.arm_qadr]))
        dq = jnp.nan_to_num(dq, nan=0.0, posinf=0.0, neginf=0.0)
        return jnp.clip(dq, -self.joint_vel_limits, self.joint_vel_limits)

    # -------------------------------------------------------- observation #
    def get_observation(self, data, visited=None):
        ee_rel = data.xpos[self.ee_body_id] - self.home_pos
        arm_q = data.qpos[self.arm_qadr]
        arm_v = data.qvel[self.arm_dofadr]
        parts = [ee_rel, arm_q, arm_v]
        if self.obs_mode == "coverage":
            cov = jnp.mean(visited) if visited is not None else jnp.float32(0.0)
            parts.append(jnp.array([cov]))
        obs = jnp.concatenate(parts)
        obs = jnp.nan_to_num(obs, nan=0.0, posinf=self.OBS_CLIP, neginf=-self.OBS_CLIP)
        return jnp.clip(obs, -self.OBS_CLIP, self.OBS_CLIP)

    def _terms(self, data, visited):
        ee_rel = data.xpos[self.ee_body_id] - self.home_pos
        r = jnp.linalg.norm(ee_rel[:2] - self.pizza_off)
        coverage = jnp.mean(visited)
        off_pizza = jnp.maximum(r - self.PIZZA_R, 0.0)
        return r, coverage, off_pizza

    def _imitation_score(self, qpos, episode_length):
        """exp(-mean squared joint-position error) against the pooled expert
        reference at this episode's progress fraction. Logging only -- never a
        reward feature, never a training signal, per the same reasoning as
        coverage (real deployment has no ground truth to score this against)."""
        progress = jnp.clip(episode_length / (self.EXPECTED_LENGTH - 1), 0.0, 1.0)
        bi = jnp.clip((progress * self.REF_N_BINS).astype(jnp.int32),
                      0, self.REF_N_BINS - 1)
        ref = self.ref_qpos[bi]
        return jnp.exp(-jnp.mean(jnp.square(qpos - ref)))

    def get_reward(self, data, action, visited):
        r, coverage, off_pizza = self._terms(data, visited)
        is_success = (coverage >= self.success_threshold).astype(jnp.float32)
        if self.reward_style == "sparse":
            reward = jnp.where(is_success > 0.5, 1.0, 0.0)
        else:
            # spread the sauce (dominant), stay on the pizza, move gently
            reward = (10.0 * coverage - 5.0 * off_pizza
                      - 0.001 * jnp.sum(jnp.square(action)))
        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=-10.0)
        info = {
            "env_info/coverage": coverage,
            "env_info/radius": r,
            "env_info/off_pizza": off_pizza,
            "env_info/is_success": is_success,
            "env_info/task_err": 1.0 - coverage,
        }
        return reward, info

    # ------------------------------------------------------------ rollout #
    @partial(jax.vmap, in_axes=(None, 0, None))
    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, eval_mode):
        data = self.mjx_data
        next_observation = jnp.zeros(self.single_observation_space.shape, dtype=jnp.float32)
        info = {
            "rollout/episode_return": 0.0,
            "rollout/episode_length": 0,
            "rollout/is_success": 0.0,
            "rollout/task_err": 0.0,
            "rollout/coverage": 0.0,
            "rollout/max_radius": 0.0,
            "rollout/off_pizza_frac": 0.0,
            "rollout/diverged": 0.0,
            "rollout/wandered_off": 0.0,
            "rollout/imitation_score": 0.0,
            "env_info/coverage": 0.0,
            "env_info/radius": 0.0,
            "env_info/off_pizza": 0.0,
            "env_info/is_success": 0.0,
            "env_info/task_err": 0.0,
            "env_info/diverged": 0.0,
            "env_info/imitation_score": 0.0,
        }
        store = {"episode_return": 0.0, "episode_length": 0,
                 "visited": jnp.zeros(self.N_RAD * self.N_ANG, dtype=jnp.float32),
                 "max_radius": 0.0, "off_steps": 0.0, "wander_steps": 0.0,
                 "imitation_score_sum": 0.0}
        state = State(data, next_observation, next_observation, 0.0, False, False,
                      info, store, key)
        return self._reset(state)

    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, state):
        data = self.mjx_data.replace(qpos=self.initial_qpos, qvel=self.initial_qvel,
                                     ctrl=jnp.zeros(self.mjx_model.nu))
        data = mjx.forward(self.mjx_model, data)
        visited = jnp.zeros(self.N_RAD * self.N_ANG, dtype=jnp.float32)
        # the tool starts near the centre (15 mm off it), so the inner ring the
        # tool disc already covers counts as swept from step 0
        visited = self._mark(visited, (data.xpos[self.ee_body_id] - self.home_pos)[:2])
        obs = self.get_observation(data, visited)
        info = dict(state.info)
        r0, cov0, off0 = self._terms(data, visited)
        for k in ("env_info/is_success", "env_info/diverged"):
            info[k] = 0.0
        # report the true starting geometry rather than zeros, so the first
        # logged point of a curve is not a fake origin
        info["env_info/radius"] = r0
        info["env_info/coverage"] = cov0
        info["env_info/off_pizza"] = off0
        info["env_info/task_err"] = 1.0 - cov0
        info["env_info/imitation_score"] = self._imitation_score(
            data.qpos[self.arm_qadr], jnp.float32(0.0))
        return state.replace(
            data=data, next_observation=obs, actual_next_observation=obs,
            reward=0.0, terminated=False, truncated=False, info=info,
            info_episode_store={"episode_return": 0.0, "episode_length": 0,
                                "visited": visited, "max_radius": 0.0, "off_steps": 0.0,
                                "wander_steps": 0.0, "imitation_score_sum": 0.0},
        )

    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        return self._step(state, action)

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action):
        vel_xy = jnp.clip(action, -1.0, 1.0) * self.MAX_SPEED

        def substep(data, _):
            dq = self._differential_ik(data, vel_xy)
            return mjx.step(self.mjx_model, data.replace(ctrl=dq)), None

        data, _ = jax.lax.scan(substep, state.data, xs=(),
                               length=self.nr_intermediate_steps)

        store = state.info_episode_store
        store["episode_length"] += 1
        xy = (data.xpos[self.ee_body_id] - self.home_pos)[:2]
        store["visited"] = self._mark(store["visited"], xy)
        visited = store["visited"]

        obs = self.get_observation(data, visited)
        reward, r_info = self.get_reward(data, action, visited)
        r = r_info["env_info/radius"]
        store["max_radius"] = jnp.maximum(store["max_radius"], r)
        store["off_steps"] += (r_info["env_info/off_pizza"] > 0.0).astype(jnp.float32)

        imitation_score = self._imitation_score(
            data.qpos[self.arm_qadr], store["episode_length"].astype(jnp.float32))
        r_info["env_info/imitation_score"] = imitation_score
        store["imitation_score_sum"] += imitation_score

        terminated = r_info["env_info/is_success"] > 0.5
        diverged = jnp.any(jnp.isnan(data.qpos)) | jnp.any(jnp.isnan(data.qvel))

        # wandered off the pizza: past the margin, held continuously for WANDER_HOLD_S
        beyond_margin = r_info["env_info/off_pizza"] > self.WANDER_MARGIN
        store["wander_steps"] = jnp.where(beyond_margin, store["wander_steps"] + 1, 0.0)
        wandered_off = store["wander_steps"] >= self.WANDER_HOLD_S * self.CONTROL_FREQ

        at_horizon = store["episode_length"] >= self.horizon
        truncated = at_horizon | diverged | wandered_off
        done = terminated | truncated

        state.info.update(r_info)
        state.info["env_info/diverged"] = diverged.astype(jnp.float32)
        store["episode_return"] += reward
        log_done = terminated | at_horizon
        for k, v in (("rollout/episode_return", store["episode_return"]),
                     ("rollout/episode_length", store["episode_length"])):
            state.info[k] = jnp.where(log_done, v, state.info[k])
        state.info["rollout/is_success"] = jnp.where(done, terminated.astype(jnp.float32), state.info["rollout/is_success"])
        state.info["rollout/task_err"] = jnp.where(done, r_info["env_info/task_err"], state.info["rollout/task_err"])
        state.info["rollout/coverage"] = jnp.where(done, r_info["env_info/coverage"], state.info["rollout/coverage"])
        state.info["rollout/max_radius"] = jnp.where(done, store["max_radius"], state.info["rollout/max_radius"])
        state.info["rollout/off_pizza_frac"] = jnp.where(
            done, store["off_steps"] / jnp.maximum(store["episode_length"], 1),
            state.info["rollout/off_pizza_frac"])
        state.info["rollout/diverged"] = jnp.where(done, diverged.astype(jnp.float32), state.info["rollout/diverged"])
        state.info["rollout/wandered_off"] = jnp.where(done, wandered_off.astype(jnp.float32), state.info["rollout/wandered_off"])
        state.info["rollout/imitation_score"] = jnp.where(
            done, store["imitation_score_sum"] / jnp.maximum(store["episode_length"], 1),
            state.info["rollout/imitation_score"])

        def when_done(_):
            s = self._reset(state)
            return s.replace(actual_next_observation=obs, reward=reward,
                             terminated=terminated, truncated=truncated)

        def when_not_done(_):
            return state.replace(data=data, next_observation=obs,
                                 actual_next_observation=obs, reward=reward,
                                 terminated=terminated, truncated=truncated)

        return jax.lax.cond(done, when_done, when_not_done, None)

    # ------------------------------------------------------------ misc #
    def render(self, state):
        data = mjx.get_data(self.mj_model, state.data)[0]
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
        squeeze = observation.ndim == 1
        if squeeze:
            observation = observation[None, :]
            action = action[None, :]

        # obs: ee_pos_rel_home 0:3 | arm_qpos 3:10 | arm_qvel 10:17 [| coverage 17]
        # NOTE obs[2] (tool HEIGHT) is deliberately never used. The real data was
        # collected under compliant hand-guiding, so height wanders with sd 8 mm,
        # while a planar sim holds it fixed -- a discriminator given that dim could
        # separate expert from policy on it alone, learning nothing about spreading.
        c = jnp.asarray(self.PIZZA_OFF, dtype=observation.dtype)
        r = jnp.linalg.norm(observation[:, 0:2] - c, axis=-1)      # radius from centre
        r_norm = r / self.PIZZA_R
        off = jnp.maximum(r - self.PIZZA_R, 0.0)
        speed = jnp.linalg.norm(observation[:, 10:17], axis=-1)    # joint-space speed
        # NO coverage term anywhere below. Coverage is a model of the swept area,
        # not something the real rig can measure, so it stays a logging metric.
        # Spreading is still expressible: the RBF bank is a radial visitation
        # histogram, and MCE-IRL matches feature EXPECTATIONS, so matching it
        # forces the policy to spend the expert's fraction of time at each radius
        # -- a policy that circles at one radius cannot match it.
        if self.feature_fn == "spread":
            # Radial occupancy at a finer resolution than base_rbf.
            bumps = [jnp.exp(-((r_norm - m) / 0.14) ** 2)
                     for m in (0.1, 0.3, 0.5, 0.7, 0.9)]
            features = jnp.stack([r_norm, -off, -speed] + bumps, axis=-1)
        elif self.feature_fn == "base_rbf":
            b1 = jnp.exp(-((r_norm - 0.25) / 0.2) ** 2)
            b2 = jnp.exp(-((r_norm - 0.55) / 0.2) ** 2)
            b3 = jnp.exp(-((r_norm - 0.85) / 0.2) ** 2)
            features = jnp.stack([r_norm, -off, -speed, b1, b2, b3], axis=-1)
        elif self.feature_fn == "state_action":
            features = jnp.concatenate([observation, action], axis=-1)
        elif self.feature_fn == "base":
            features = jnp.stack([r_norm, -off, -speed], axis=-1)
        elif self.feature_fn == "learned":
            features = self.learned_features(observation)
        else:
            features = observation

        return features[0] if squeeze else features
