"""Assemble the real bulb-slotting demonstrations into a TRIRL training npz for
the 2-D action-space variant (bulbslot2d_mjx).

Reuses bulbslot_mjx's real recordings (25-D states, seat frame, identical
observation layout -- the plane restriction changes nothing about what's
OBSERVED) but relabels the actions from scratch for the reduced 3-D action
space: in-plane velocity (u, z) + one binary gripper bit, instead of the
original free 3-D velocity + wrist yaw + continuous gripper width.

1. RELABEL LINEAR: forward-kinematics the recorded joint encoders to the world
   EE path (same v2 method as the other bulb tasks -- forward difference, not
   centred, since the label at t must be the action that produced t -> t+1),
   then project that velocity onto the SAME plane basis (u_hat, z_hat, n_hat)
   the environment derives from the scene. The out-of-plane (n) component is
   thrown away -- it's what the plane restriction gives up -- and reported so
   its size is visible, not hidden.

2. RELABEL GRIPPER as a single binary bit, not from the raw recorded action
   (whose gripper convention turned out not to have a clean threshold at 0 --
   it never commands a fully-closed target, because the jaws stall against the
   60 mm bulb head at ~58 mm, not 0) but from the STATE's own width channel,
   using the exact criterion the environment's _grip_weld uses: width >= 65 mm
   -> open (+1), else closed (-1). This guarantees the relabelled action
   reproduces the demonstrated grasp transition when replayed through the
   env's own weld logic, by construction.

3. CANONICALISE the bulb quaternion to its swing (unchanged from bulbslot_mjx
   -- state-only, doesn't depend on the action space at all).

4. TRUNCATE at depth-AND-released, held 1s -- NOT depth alone. This env's
   success criterion requires the release (see environment.py), so the
   training data must include it: truncating at depth alone (as
   build_slot_training_set.py does for the full 3-D env) would cut every
   episode before the release ever happens, and the discriminator would never
   see the one event this action space exists to produce.

Usage:
    python scripts/build_bulbslot2d_training_set.py
"""
import argparse
from pathlib import Path

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "trust_region_irl" / "environments" / "bulbslot2d_mjx" / "data" / "scene_mjx_slot.xml"
DT = 0.05
DEPTH_SOLVED, HOLD_S = 0.020, 1.0
GRIP_CLOSED = 0.065


def smooth(x, k):
    if k <= 1:
        return x
    pad = k // 2
    xp = np.pad(x, [(pad, pad)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    ker = np.ones(k) / k
    return np.stack([np.convolve(xp[:, i], ker, mode="valid")[:len(x)]
                     for i in range(x.shape[1])], axis=1)


def fwd_diff(x, dt):
    v = np.diff(x, axis=0) / dt
    return np.concatenate([v, v[-1:]], axis=0)


def swing_quat(q, eps=1e-9):
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


def first_slotted_released(d_seat, width, thr=DEPTH_SOLVED, hold=HOLD_S, dt=DT):
    n = int(round(hold / dt))
    ok = (d_seat < thr) & (width >= GRIP_CLOSED)
    for i in range(len(ok) - n + 1):
        if ok[i:i + n].all():
            return i + n - 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path,
                    default=REPO / "trirl_dataset" / "real_expert" / "slot_raw_2026-09-06")
    ap.add_argument("--out", type=Path,
                    default=REPO / "trirl_dataset" / "rl_expert" / "expert_dataset_bulbslot2d_real.npz")
    ap.add_argument("--relabelled-dir", type=Path,
                    default=REPO / "trirl_dataset" / "real_expert" / "slot2d_relabelled")
    ap.add_argument("--max-speed", type=float, default=0.10)
    ap.add_argument("--smooth", type=int, default=5)
    args = ap.parse_args()

    from trust_region_irl.environments.bulbslot2d_mjx.environment import BulbSlot2D
    env = BulbSlot2D(render=False, feature_fn="base_slot")
    u_hat = np.asarray(env.u_hat); z_hat = np.asarray(env.z_hat); n_hat = np.asarray(env.n_hat)
    print(f"plane basis: u_hat {np.round(u_hat,3)}  n_hat {np.round(n_hat,3)}")

    m = mujoco.MjModel.from_xml_path(MODEL.as_posix())
    d = mujoco.MjData(m)
    jadr = [m.joint(f"fr3_joint{i+1}").qposadr[0] for i in range(7)]
    ee = m.body("ee_frame").id

    args.relabelled_dir.mkdir(parents=True, exist_ok=True)
    eps = sorted(args.in_dir.glob("*.npz"))
    S, A, NS, AB, R = [], [], [], [], []
    lens, dropped, n_resid, clip_frac = [], [], [], []
    print(f"{len(eps)} episodes | max_speed {args.max_speed} m/s")
    print(f"{'episode':<46} {'full':>5} {'kept':>5} {'cut@':>5}")
    print("-" * 66)
    for p in eps:
        z = dict(np.load(p))
        st, ac = z["states"], z["actions"]

        pos = np.empty((len(st), 3))
        for i, q in enumerate(st[:, 11:18]):
            d.qpos[jadr] = q
            mujoco.mj_forward(m, d)
            pos[i] = d.xpos[ee]
        v = smooth(fwd_diff(pos, DT), args.smooth)
        v_u = v @ u_hat
        v_z = v @ z_hat
        v_n = v @ n_hat
        n_resid.append(np.abs(v_n))

        new = np.zeros((len(st), 3), dtype=np.float32)
        new[:, 0] = np.clip(v_u / args.max_speed, -1.0, 1.0)
        new[:, 1] = np.clip(v_z / args.max_speed, -1.0, 1.0)
        width = st[:, 10]
        new[:, 2] = np.where(width >= GRIP_CLOSED, 1.0, -1.0)
        clip_frac.append(np.mean((np.abs(v_u) > args.max_speed) | (np.abs(v_z) > args.max_speed)))

        st2 = st.copy(); ns2 = z["next_states"].copy()
        st2[:, 3:7] = swing_quat(st[:, 3:7])
        ns2[:, 3:7] = swing_quat(z["next_states"][:, 3:7])

        np.savez(args.relabelled_dir / p.name, states=st2.astype(np.float32),
                 actions=new.astype(np.float32), next_states=ns2.astype(np.float32),
                 absorbing=z["absorbing"], rewards=z["rewards"])

        d_seat = np.linalg.norm(st2[:, 0:3], axis=1)
        cut = first_slotted_released(d_seat, width)
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
             rewards=np.concatenate(R).astype(np.float32),
             episode_lengths=np.array(lens, dtype=np.int32))
    lens = np.array(lens)
    tot = sum(len(np.load(p)["states"]) for p in eps)
    n_resid = np.concatenate(n_resid)
    print("-" * 66)
    print(f"{len(lens)} episodes, {lens.sum()} transitions ({lens.sum()*DT:.0f} s at 20 Hz)")
    print(f"lengths: min {lens.min()} median {int(np.median(lens))} max {lens.max()}")
    print(f"kept {100*lens.sum()/tot:.0f}% of the recorded transitions; {len(dropped)} dropped")
    print(f"out-of-plane speed discarded: median {1000*np.median(n_resid):.1f} mm/s, "
          f"p90 {1000*np.percentile(n_resid,90):.1f} mm/s")
    print(f"action clipped (|v_u| or |v_z| > max_speed): {100*np.mean(clip_frac):.1f}% of steps")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
