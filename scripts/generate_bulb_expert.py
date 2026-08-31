"""Generate bulb-screwing expert trajectories with a scripted FSM controller.

The FSM (ported from FurnitureBench's lamp_bulb assembly logic, adapted to the
bulbscrew_mjx action space) servoes the ee through:
  reach_above -> descend -> grasp (width-confirmed) -> settle -> lift ->
  carry -> insert  ... the env terminates on success (bulb seated + upright).

Runs the env's own step (MJX), so demos are recorded exactly in the training
observation/action space (25-D, matching the real rig). Only successes are kept.

Output npz matches the pushT expert format:
    states      (T, 25)   env observation
    actions     (T, 5)    env action in [-1, 1]
    next_states (T, 25)
    rewards     (T,)      env dense reward
    absorbing   (T,) f32  1.0 on the final (success) transition
    qpos        (T, 16)   exact MuJoCo pose (rendering / re-derivation)

Run in the `trirl` env:
    python scripts/generate_bulb_expert.py --num-success 100
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
import argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from mujoco import mjx
from tqdm import tqdm

from trust_region_irl.environments.bulbscrew_mjx.environment import BulbScrew
from trust_region_irl.environments.bulbscrew_mjx.general_properties import GeneralProperties

REPO = Path(__file__).resolve().parents[1]


def axis_z_of(quat):
    w, x, y, z = quat
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


class BulbFSM:
    """Per-episode scripted controller emitting 5-D env actions from ground truth."""

    def __init__(self, env, rng):
        self.env = env
        self.rng = rng
        self.state = "reach_above"
        self.t_in_state = 0
        self.failed = False
        self.noise = 0.05          # action-space exploration noise (demo diversity)

    def _measure(self, data, i):
        ee = np.asarray(data.xpos[i, self.env.ee_body_id])
        bulb = np.asarray(data.xpos[i, self.env.bulb_body_id])
        quat = np.asarray(data.xquat[i, self.env.bulb_body_id])
        az = axis_z_of(quat)
        neck = bulb + az * self.env.NECK_OFF
        tip = bulb + az * self.env.TIP_OFF
        seat = np.asarray(self.env.seat_pos)
        return ee, neck, tip, seat

    def act(self, data, obs, i=0):
        ee, neck, tip, seat = self._measure(data, i)
        # No tactile in the 25-D observation (real hand has none). Grasp is
        # confirmed by the fingers closing onto the head instead of shutting.
        width = float(obs[10])                    # total opening, 0..0.08
        a = np.zeros(5, np.float32)
        kp = 4.0

        def servo(target, vmax=1.0):
            v = kp * (np.asarray(target) - ee) / self.env.MAX_SPEED
            a[:3] = np.clip(v, -vmax, vmax)

        s = self.state
        if s == "reach_above":
            servo([neck[0], neck[1], neck[2] + 0.16]); a[4] = 1.0
            if np.linalg.norm(ee[:2] - neck[:2]) < 0.01 and abs(ee[2] - neck[2] - 0.16) < 0.02:
                self._next("descend")
        elif s == "descend":
            servo(neck, vmax=0.28); a[4] = 1.0
            if np.linalg.norm(ee - neck) < 0.008:
                self._next("grasp")
            self._timeout(200)
        elif s == "grasp":
            servo(neck, vmax=0.1); a[4] = -1.0
            # head is 60 mm: fingers stop at ~0.060 instead of closing to 0
            if width < 0.068:
                self._next("settle")
            self._timeout(60)
        elif s == "settle":
            servo(neck, vmax=0.1); a[4] = -1.0
            if self.t_in_state > 30:
                self._next("lift")
        elif s == "lift":
            # gentle initial lift (demo-tuned: a fast lift slips the grip)
            servo([ee[0], ee[1], 0.25], vmax=0.17 if ee[2] < 0.13 else 0.6); a[4] = -1.0
            if ee[2] > 0.23:
                self._next("carry")
            if width > 0.075 and self.t_in_state > 15:
                self.failed = True
            self._timeout(250)
        elif s == "carry":
            # target so the TIP lands over the seat (correct for grasp offset)
            off = ee - tip
            servo([seat[0] + off[0], seat[1] + off[1], 0.25]); a[4] = -1.0
            if np.linalg.norm((tip - seat)[:2]) < 0.006:
                self._next("insert")
            if width > 0.075:
                self.failed = True
            self._timeout(200)
        elif s == "insert":
            off = ee - tip
            servo([seat[0] + off[0], seat[1] + off[1], seat[2] + off[2] + 0.004], vmax=0.22)
            a[4] = -1.0
            a[3] = 0.4            # gentle screwing twist while descending
            if width > 0.075:
                self.failed = True
            self._timeout(300)
        # exploration noise on the motion channels only (not the gripper)
        a[:4] = np.clip(a[:4] + self.rng.normal(0, self.noise, 4), -1.0, 1.0)
        self.t_in_state += 1
        return a

    def _next(self, s):
        self.state = s
        self.t_in_state = 0

    def _timeout(self, n):
        if self.t_in_state > n:
            self.failed = True


def jitter_start(env, state, rng):
    """Small start variation on every slot: bulb xy (holder slack) + arm pose."""
    data = state.data
    qpos = np.array(data.qpos)  # copy: np.asarray on a jax array is read-only
    n = qpos.shape[0]
    qpos[:, env.bulb_qadr:env.bulb_qadr + 2] += rng.uniform(-0.0015, 0.0015, (n, 2))
    arm_adr = np.asarray(env.arm_qadr)
    qpos[:, arm_adr] += rng.uniform(-0.01, 0.01, (n, 7))
    data = data.replace(qpos=jnp.asarray(qpos, dtype=jnp.float32))
    data = jax.vmap(lambda d: mjx.forward(env.mjx_model, d))(data)
    obs = jax.vmap(env.get_observation)(data)
    return state.replace(data=data, next_observation=obs, actual_next_observation=obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-success", type=int, default=100)
    ap.add_argument("--max-candidates", type=int, default=400)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16, help="parallel envs, one FSM each")
    ap.add_argument("--out", type=Path,
                    default=REPO / "trirl_dataset" / "rl_expert" / "expert_dataset_bulbscrew_fsm.npz")
    args = ap.parse_args()

    env = BulbScrew(render=False, feature_fn="base")
    env.general_properties = GeneralProperties
    N = args.batch

    key = jax.random.PRNGKey(args.seed_start)
    state = env.reset(jax.random.split(key, N), False)
    state = jitter_start(env, state, np.random.default_rng(args.seed_start))

    seed_ctr = args.seed_start
    def new_fsm():
        nonlocal seed_ctr
        seed_ctr += 1
        return BulbFSM(env, np.random.default_rng(seed_ctr))

    fsms = [new_fsm() for _ in range(N)]
    bufs = [{k: [] for k in "sanrq"} for _ in range(N)]

    S, A, NS, R, AB, Q = [], [], [], [], [], []
    n_ok = attempts = 0
    fail_hist = {}
    pbar = tqdm(total=args.num_success, desc="successes", unit="ep", dynamic_ncols=True)
    while n_ok < args.num_success and attempts < args.max_candidates:
        obs_all = np.asarray(state.next_observation)
        qpos_all = np.asarray(state.data.qpos, dtype=np.float32)
        acts = np.stack([fsms[i].act(state.data, obs_all[i], i) for i in range(N)])
        state = env.step(state, jnp.asarray(acts, dtype=jnp.float32))
        ns_all = np.asarray(state.actual_next_observation)
        r_all = np.asarray(state.reward)
        term = np.asarray(state.terminated)
        trunc = np.asarray(state.truncated)

        for i in range(N):
            b = bufs[i]
            b["s"].append(obs_all[i]); b["a"].append(acts[i]); b["n"].append(ns_all[i])
            b["r"].append(float(r_all[i])); b["q"].append(qpos_all[i])
            if term[i] or trunc[i]:
                attempts += 1
                if not term[i]:
                    k = fsms[i].state + ("(failed)" if fsms[i].failed else "")
                    fail_hist[k] = fail_hist.get(k, 0) + 1
                if term[i] and n_ok < args.num_success:   # success -> harvest
                    absorbing = np.zeros(len(b["s"]), np.float32); absorbing[-1] = 1.0
                    S.append(np.array(b["s"], np.float32)); A.append(np.array(b["a"], np.float32))
                    NS.append(np.array(b["n"], np.float32)); R.append(np.array(b["r"], np.float32))
                    AB.append(absorbing); Q.append(np.array(b["q"], np.float32))
                    n_ok += 1
                    pbar.update(1)
                bufs[i] = {k: [] for k in "sanrq"}
                fsms[i] = new_fsm()   # env slot auto-reset itself
                rate = 100.0 * n_ok / max(attempts, 1)
                pbar.set_postfix_str(f"attempts={attempts} success_rate={rate:.0f}%")
    pbar.close()
    print("failure histogram (FSM state at episode end):", dict(sorted(fail_hist.items(), key=lambda x: -x[1])))

    if n_ok == 0:
        print("No successful trajectories generated.")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out,
             states=np.concatenate(S), actions=np.concatenate(A),
             next_states=np.concatenate(NS), rewards=np.concatenate(R),
             absorbing=np.concatenate(AB), qpos=np.concatenate(Q))
    lens = [len(x) for x in S]
    print(f"Saved {n_ok} episodes ({sum(lens)} transitions) -> {args.out}")
    print(f"success rate: {n_ok}/{attempts}; episode length min/mean/max = "
          f"{min(lens)}/{np.mean(lens):.1f}/{max(lens)}")


if __name__ == "__main__":
    main()
