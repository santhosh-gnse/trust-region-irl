"""Assemble the real bulb-SLOTTING demonstrations into one TRIRL training npz.

Four things happen here.

1. RELABEL the actions (the "v2" scheme). Kinesthetic demonstrations have no
   commanded action, so actions are derived from the motion the operator
   produced, by forward kinematics on the recorded joint encoders. The
   derivative is a FORWARD difference -- the label at t must be the action that
   produced the transition to t+1, and np.gradient's centred difference partly
   encodes motion that had already happened.

   Scales stay at the rig's 0.10 m/s and 0.75 rad/s, unlike the screwing set
   which needed achieved rates. Measured here: yaw exceeds its cap on 0.1% of
   steps (nothing is threaded) and the linear axes on 5.4%, with p95 at
   0.104 m/s -- essentially the cap. Raising them would buy little and cost
   coarser exploration on a task whose whole difficulty is the final few mm.

2. CANONICALISE the bulb quaternion to its swing (tilt only, no rotation about
   the bulb's own axis). The real axis-yaw sits at 107.6 +- 2.5 deg, set by how
   the mocap rigid body was defined, while the sim has a single fixed value --
   a systematic 30 deg difference on a quantity that says nothing about a task
   where nothing is threaded. Left in, a discriminator can separate expert from
   policy on that alone. bulbslot_mjx applies the same transform, so both sides
   agree. Tilt is preserved exactly.

3. TRUNCATE at success: the first step where d_seat < 20 mm and stays under for
   1 s, which is where a bulbslot_mjx episode terminates. Recorded episodes
   continue past that point while the operator withdraws.

4. CONCATENATE into a single npz with absorbing = 1.0 on each episode's final
   step, float32 (a bool array trips a cast bug downstream).

Usage:
    MUJOCO_GL=egl python scripts/build_slot_training_set.py
"""
import argparse
from pathlib import Path

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "trust_region_irl" / "environments" / "bulbslot_mjx" / "data" / "scene_mjx_slot.xml"
DT = 0.05
DEPTH_SOLVED, HOLD_S = 0.020, 1.0
OLD_LINEAR, OLD_YAW = 0.10, 0.75          # scales the recorder used


def smooth(x, k):
    if k <= 1:
        return x
    pad = k // 2
    xp = np.pad(x, [(pad, pad)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    ker = np.ones(k) / k
    return np.stack([np.convolve(xp[:, i], ker, mode="valid")[:len(x)]
                     for i in range(x.shape[1])], axis=1)


def fwd_diff(x, dt):
    """Forward difference: the rate that produced s[t] -> s[t+1]."""
    v = np.diff(x, axis=0) / dt
    return np.concatenate([v, v[-1:]], axis=0)


def swing_quat(q, eps=1e-9):
    """Strip rotation about the body's own axis, keeping the tilt exactly."""
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + eps)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    az = np.stack([2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)], axis=-1)
    az = az / (np.linalg.norm(az, axis=-1, keepdims=True) + eps)
    cos_t = np.clip(az[..., 2], -1.0, 1.0)
    axis = np.stack([-az[..., 1], az[..., 0], np.zeros_like(cos_t)], axis=-1)
    n = np.linalg.norm(axis, axis=-1, keepdims=True)
    u = np.where(n > eps, axis / (n + eps), np.array([1.0, 0.0, 0.0]))
    half = 0.5 * np.arccos(cos_t)[..., None]
    sw = np.concatenate([np.cos(half), np.sin(half) * u], axis=-1)
    return sw / (np.linalg.norm(sw, axis=-1, keepdims=True) + eps)


def first_slotted(d_seat, thr=DEPTH_SOLVED, hold=HOLD_S, dt=DT):
    n = int(round(hold / dt))
    under = d_seat < thr
    for i in range(len(under) - n + 1):
        if under[i:i + n].all():
            return i + n - 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path,
                    default=REPO / "trirl_dataset" / "real_expert" / "slot_raw_2026-09-06")
    ap.add_argument("--out", type=Path,
                    default=REPO / "trirl_dataset" / "rl_expert" / "expert_dataset_bulbslot_real.npz")
    ap.add_argument("--relabelled-dir", type=Path,
                    default=REPO / "trirl_dataset" / "real_expert" / "slot_relabelled_v2")
    ap.add_argument("--max-linear-speed", type=float, default=0.10)
    ap.add_argument("--max-yaw-rate", type=float, default=0.75)
    ap.add_argument("--smooth", type=int, default=5)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(MODEL.as_posix())
    d = mujoco.MjData(m)
    jadr = [m.joint(f"fr3_joint{i+1}").qposadr[0] for i in range(7)]
    ee = m.body("ee_frame").id

    args.relabelled_dir.mkdir(parents=True, exist_ok=True)
    eps = sorted(args.in_dir.glob("*.npz"))
    S, A, NS, AB, R = [], [], [], [], []
    lens, dropped, res_l, res_y = [], [], [], []
    print(f"{len(eps)} episodes | scales {args.max_linear_speed} m/s, {args.max_yaw_rate} rad/s")
    print(f"{'episode':<46} {'full':>5} {'kept':>5} {'cut@':>5}")
    print("-" * 66)
    for p in eps:
        z = dict(np.load(p))
        st, ac = z["states"], z["actions"]

        # --- 1. relabel from the encoders, forward difference ---
        pos = np.empty((len(st), 3)); yaw = np.empty(len(st))
        for i, q in enumerate(st[:, 11:18]):
            d.qpos[jadr] = q
            mujoco.mj_forward(m, d)
            pos[i] = d.xpos[ee]
            Rm = d.xmat[ee].reshape(3, 3)
            yaw[i] = np.arctan2(Rm[1, 0], Rm[0, 0])
        yaw = np.unwrap(yaw)
        v = smooth(fwd_diff(pos, DT), args.smooth)
        w = smooth(fwd_diff(yaw[:, None], DT), args.smooth)[:, 0]
        new = ac.copy()
        new[:, 0:3] = np.clip(v / args.max_linear_speed, -1.0, 1.0)
        new[:, 3] = np.clip(w / args.max_yaw_rate, -1.0, 1.0)
        # action[4] (gripper) is a width, not a rate -> unchanged

        # validation against the unsaturated original labels
        ul = np.all(np.abs(ac[:, 0:3]) < 0.99, axis=1)
        uy = np.abs(ac[:, 3]) < 0.99
        if ul.any():
            res_l.append(np.median(np.abs(new[ul, 0:3]*args.max_linear_speed - ac[ul, 0:3]*OLD_LINEAR)))
        if uy.any():
            res_y.append(np.median(np.abs(new[uy, 3]*args.max_yaw_rate - ac[uy, 3]*OLD_YAW)))

        # --- 2. canonicalise the bulb quaternion (both states and next_states) ---
        st2 = st.copy(); ns2 = z["next_states"].copy()
        st2[:, 3:7] = swing_quat(st[:, 3:7])
        ns2[:, 3:7] = swing_quat(z["next_states"][:, 3:7])

        np.savez(args.relabelled_dir / p.name, states=st2.astype(np.float32),
                 actions=new.astype(np.float32), next_states=ns2.astype(np.float32),
                 absorbing=z["absorbing"], rewards=z["rewards"])

        # --- 3. truncate at slotted ---
        d_seat = np.linalg.norm(st2[:, 0:3], axis=1)
        cut = first_slotted(d_seat)
        if cut is None:
            dropped.append(p.stem)
            print(f"{p.stem:<46} {len(st):>5} {'DROP':>5} {'never':>5}")
            continue
        n = cut + 1
        ab = np.zeros(n, np.float32); ab[-1] = 1.0
        S.append(st2[:n]); A.append(new[:n]); NS.append(ns2[:n])
        AB.append(ab); R.append(np.zeros(n, np.float32))
        lens.append(n)
        print(f"{p.stem:<46} {len(st):>5} {n:>5} {cut:>5}", flush=True)

    if not S:
        raise SystemExit("no episodes kept")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out,
             states=np.concatenate(S).astype(np.float32),
             actions=np.concatenate(A).astype(np.float32),
             next_states=np.concatenate(NS).astype(np.float32),
             absorbing=np.concatenate(AB).astype(np.float32),
             rewards=np.concatenate(R).astype(np.float32))
    lens = np.array(lens)
    tot = sum(len(np.load(p)["states"]) for p in eps)
    print("-" * 66)
    print(f"{len(lens)} episodes, {lens.sum()} transitions ({lens.sum()*DT:.0f} s at 20 Hz)")
    print(f"lengths: min {lens.min()} median {int(np.median(lens))} max {lens.max()}")
    print(f"kept {100*lens.sum()/tot:.0f}% of the recorded transitions; {len(dropped)} dropped")
    print(f"relabel validation on unsaturated steps: linear {1000*np.median(res_l):.2f} mm/s, "
          f"yaw {np.median(res_y):.3f} rad/s")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
