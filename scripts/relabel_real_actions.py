"""Recompute the action labels of real kinesthetic episodes at a new action scale.

WHY. Actions in a kinesthetic demonstration are not commanded, they are DERIVED
from the motion the operator produced, by inverting what safety_node applies
forward (kinesthetic_recorder_node):

    action[0:3] = ee_velocity / max_linear_speed     (in fr3_link0)
    action[3]   = wrist yaw rate / max_yaw_rate
    action[4]   = gripper term

then clipped to [-1, 1]. If a scale is too small the label saturates and the
excess is thrown away: measured on this dataset the wrist exceeded the recorded
`max_yaw_rate = 0.75` rad/s on 64% of turning steps, median 1.32 rad/s.

The thrown-away part is RECOVERABLE without re-recording, because the motion
itself survives in the observation: `arm_qpos` (obs[11:18]) are joint encoder
readings, so forward kinematics gives the tool pose, and differencing gives the
velocity the operator actually produced -- cleanly, with no mocap involved.
This script rebuilds the labels at a new scale from that.

WHAT IT DOES NOT TOUCH. The gripper channel (action[4]) is copied unchanged:
its forward mapping is a width, not a rate, so no scale correction applies.
Observations are copied unchanged. Originals are never modified.

VALIDATION. On steps that were NOT saturated the original label is trustworthy,
so `new_action * new_scale` must reproduce `old_action * old_scale`. The script
reports that residual per channel; a large one means the FK reconstruction does
not match the recorder and the output should not be trusted.

Usage:
    MUJOCO_GL=egl python scripts/relabel_real_actions.py \
        --in-dir  trirl_dataset/real_expert/raw_2026-08-30 \
        --out-dir trirl_dataset/real_expert/relabelled \
        --max-linear-speed 0.10 --max-yaw-rate 2.6
"""
import argparse
from pathlib import Path

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "trust_region_irl" / "environments" / "bulbscrew_mjx" / "data" / "scene_mjx_bulb.xml"
DT = 0.05                     # 20 Hz control
OLD_LINEAR, OLD_YAW = 0.10, 0.75      # scales in force when these were recorded


def fwd_diff(x, dt):
    """Forward difference: the rate that produced the transition s[t] -> s[t+1].

    np.gradient uses CENTRED differences, so a[t] would partly encode motion
    between t-1 and t -- motion that had already happened. The label is supposed
    to be the action that CAUSED the next state, which is the forward difference.
    Measured on this data the two disagree by 2.2% of the speed cap at the median
    and 28.5% at the 90th percentile, so it is small but not negligible.
    The final step has no successor; it repeats the previous rate.
    """
    v = np.diff(x, axis=0) / dt
    return np.concatenate([v, v[-1:]], axis=0)


def smooth(x, k):
    """Centred moving average along axis 0; k=1 is a no-op."""
    if k <= 1:
        return x
    pad = k // 2
    xp = np.pad(x, [(pad, pad)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    ker = np.ones(k) / k
    return np.stack([np.convolve(xp[:, i], ker, mode="valid")[:len(x)]
                     for i in range(x.shape[1])], axis=1)


def tool_pose_series(S, m, d, jadr, ee):
    """Tool position and yaw about world z, by FK from the joint encoders."""
    pos = np.empty((len(S), 3))
    yaw = np.empty(len(S))
    for i, q in enumerate(S[:, 11:18]):
        d.qpos[jadr] = q
        mujoco.mj_forward(m, d)
        pos[i] = d.xpos[ee]
        R = d.xmat[ee].reshape(3, 3)
        yaw[i] = np.arctan2(R[1, 0], R[0, 0])
    return pos, np.unwrap(yaw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-linear-speed", type=float, default=0.10)
    ap.add_argument("--max-yaw-rate", type=float, default=2.6)
    ap.add_argument("--smooth", type=int, default=3,
                    help="moving-average window on the derived velocities; hand "
                         "motion is noisy at 20 Hz and raw differences label the "
                         "expert as jittering (the recorder low-pass filters too)")
    ap.add_argument("--dry-run", action="store_true", help="measure only, write nothing")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(MODEL.as_posix())
    d = mujoco.MjData(m)
    jadr = [m.joint(f"fr3_joint{i+1}").qposadr[0] for i in range(7)]
    ee = m.body("ee_frame").id

    eps = sorted(args.in_dir.glob("*.npz"))
    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(eps)} episodes | new scales: linear {args.max_linear_speed} m/s, "
          f"yaw {args.max_yaw_rate} rad/s  (old: {OLD_LINEAR}, {OLD_YAW})\n")

    hdr = (f"{'episode':<50} {'lin sat%':>8} {'yaw sat%':>8} -> "
           f"{'lin sat%':>8} {'yaw sat%':>8} {'resid lin':>10} {'resid yaw':>10}")
    print(hdr); print("-" * len(hdr))
    agg = {"old_lin": [], "old_yaw": [], "new_lin": [], "new_yaw": [],
           "res_lin": [], "res_yaw": [], "v": [], "w": []}

    for p in eps:
        z = dict(np.load(p))
        S, A = z["states"], z["actions"]
        pos, yaw = tool_pose_series(S, m, d, jadr, ee)
        v = smooth(fwd_diff(pos, DT), args.smooth)                     # m/s, fr3_link0
        w = smooth(fwd_diff(yaw[:, None], DT), args.smooth)[:, 0]      # rad/s

        new = A.copy()
        new[:, 0:3] = np.clip(v / args.max_linear_speed, -1.0, 1.0)
        new[:, 3] = np.clip(w / args.max_yaw_rate, -1.0, 1.0)
        # action[4] (gripper) is a width, not a rate -> copied unchanged

        # --- validation: on unsaturated steps the old label is trustworthy, so
        # the two reconstructions must agree in PHYSICAL units.
        un_l = np.all(np.abs(A[:, 0:3]) < 0.99, axis=1)
        un_y = np.abs(A[:, 3]) < 0.99
        res_l = (np.median(np.abs(new[un_l, 0:3] * args.max_linear_speed
                                  - A[un_l, 0:3] * OLD_LINEAR)) if un_l.any() else np.nan)
        res_y = (np.median(np.abs(new[un_y, 3] * args.max_yaw_rate
                                  - A[un_y, 3] * OLD_YAW)) if un_y.any() else np.nan)

        ol = 100 * np.mean(np.any(np.abs(A[:, 0:3]) > 0.99, axis=1))
        oy = 100 * np.mean(np.abs(A[:, 3]) > 0.99)
        nl = 100 * np.mean(np.any(np.abs(new[:, 0:3]) > 0.99, axis=1))
        ny = 100 * np.mean(np.abs(new[:, 3]) > 0.99)
        for k, val in [("old_lin", ol), ("old_yaw", oy), ("new_lin", nl), ("new_yaw", ny),
                       ("res_lin", res_l), ("res_yaw", res_y)]:
            agg[k].append(val)
        agg["v"].append(np.linalg.norm(v, axis=1)); agg["w"].append(np.abs(w))

        if not args.dry_run:
            z["actions"] = new.astype(np.float32)
            np.savez(args.out_dir / p.name, **z)
        print(f"{p.stem:<50} {ol:8.1f} {oy:8.1f} -> {nl:8.1f} {ny:8.1f} "
              f"{res_l*1000:9.2f}m {res_y:10.3f}", flush=True)

    print("-" * len(hdr))
    v = np.concatenate(agg["v"]); w = np.concatenate(agg["w"])
    print(f"saturation  linear {np.mean(agg['old_lin']):.1f}% -> {np.mean(agg['new_lin']):.1f}%   "
          f"yaw {np.mean(agg['old_yaw']):.1f}% -> {np.mean(agg['new_yaw']):.1f}%")
    print(f"validation residual (unsaturated steps): linear "
          f"{1000*np.nanmedian(agg['res_lin']):.2f} mm/s, yaw {np.nanmedian(agg['res_yaw']):.3f} rad/s")
    print(f"\nachieved speeds:  linear p95 {np.percentile(v,95):.3f} m/s (cap "
          f"{args.max_linear_speed})   yaw p95 {np.percentile(w,95):.2f} rad/s "
          f"(cap {args.max_yaw_rate})")
    if not args.dry_run:
        print(f"\nwrote {len(eps)} relabelled episodes -> {args.out_dir}")
        print("Set MAX_LINEAR_SPEED/MAX_YAW_RATE in bulbscrew_mjx to the SAME values, "
              "or the labels mean different speeds on each side.")


if __name__ == "__main__":
    main()
