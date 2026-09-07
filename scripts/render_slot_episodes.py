"""Batch-replay real bulb-SLOTTING episodes in the bulbslot_mjx scene.

Real episodes carry no qpos, only the 25-D observation, so the pose is
reconstructed:

    arm joints  <- obs[11:18]        (joint encoders)
    fingers     <- obs[10] / 2 each
    bulb        <- tip = SEAT + R_L0_SEAT @ obs[0:3],  quat = Q_L0_SEAT * obs[3:7]
                   body origin = tip - axis_z * TIP_OFF

THE FRAME TRAP. obs[0:3] and obs[3:7] are in the SEAT frame. For the 2026-09-06
slotting episodes that frame is rotated -77.7 deg about z relative to fr3_link0 --
NOT the 150.7 deg of the earlier screwing session, because the socket was moved
between them. Using the wrong rotation puts the bulb ~0.4 m away, somewhere
entirely plausible, with no error raised.

The replay doubles as an alignment check: the arm comes from the encoders and the
bulb from mocap, so while the jaws are closed the reconstructed grasp point
should sit near the FK tool position. Expect ~8.5 mm here -- the manifest
documents a consistent grasp-offset calibration bias of that size affecting all
126 episodes equally.

Usage:
    MUJOCO_GL=egl python scripts/render_slot_episodes.py --data-dir <episodes/>
"""
import argparse
from pathlib import Path

import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "trust_region_irl" / "environments" / "bulbslot_mjx" / "data" / "scene_mjx_slot.xml"

TIP_OFF = -0.053
SEAT = np.array([0.6084, 0.0787, 0.0685])       # measured 2026-09-06
SEAT_OFF = np.array([0.0, 0.0, 0.018])
# seat frame -> fr3_link0 for the 2026-09-06 slotting session (-77.7 deg about z)
Q_L0_SEAT = np.array([0.778551, -0.000876, -0.003338, -0.627571])   # wxyz
DEPTH_SOLVED, HOLD_S, RATE = 0.020, 1.0, 20     # slotted: 20 mm held 1 s
GRIP_CLOSED = 0.065


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def axis_z_of(q):
    w, x, y, z = q
    return np.array([2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)])


def first_slotted(d_seat, thr=DEPTH_SOLVED, hold=HOLD_S, rate=RATE):
    n = int(round(hold * rate))
    under = d_seat < thr
    for i in range(len(under) - n + 1):
        if under[i:i + n].all():
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=REPO / "slot_episode_videos")
    ap.add_argument("--fps", type=int, default=40)          # 2x real time
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=608)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(MODEL.as_posix())
    m.vis.global_.offwidth, m.vis.global_.offheight = args.width, args.height
    d = mujoco.MjData(m)

    seat_in_model = m.body_pos[m.body("socket").id] + SEAT_OFF
    if not np.allclose(seat_in_model, SEAT, atol=1e-3):
        raise SystemExit(f"scene seat {seat_in_model} != {SEAT}; wrong scene?")

    R_L0_SEAT = quat_to_mat(Q_L0_SEAT)
    jadr = [m.joint(f"fr3_joint{i+1}").qposadr[0] for i in range(7)]
    f1 = m.joint("finger_joint1").qposadr[0]
    f2 = m.joint("finger_joint2").qposadr[0]
    bq = m.joint("bulb_joint").qposadr[0]
    ee_bid = m.body("ee_frame").id

    r = mujoco.Renderer(m, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.67, -0.05, 0.12]
    cam.distance, cam.elevation, cam.azimuth = 0.95, -22, 150

    args.outdir.mkdir(exist_ok=True)
    eps = sorted(args.data_dir.glob("*.npz"))
    print(f"{len(eps)} episodes -> {args.outdir}\n")
    hdr = f"{'episode':<46} {'steps':>6} {'final d':>8} {'tilt':>7} {'slot@':>7} {'grasp':>7}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for p in eps:
        out = args.outdir / f"{p.stem}.mp4"
        if args.skip_existing and out.exists():
            continue
        S = np.load(p)["states"]
        d_seat = np.linalg.norm(S[:, 0:3], axis=1)
        q_raw = S[:, 3:7] / (np.linalg.norm(S[:, 3:7], axis=1, keepdims=True) + 1e-9)
        q_raw = np.where(q_raw[:, :1] < 0, -q_raw, q_raw)
        tilt = np.degrees(np.arccos(np.clip(1 - 2*(q_raw[:, 1]**2 + q_raw[:, 2]**2), -1, 1)))
        grasp_err = np.linalg.norm(S[:, 7:10], axis=1)
        gripped = S[:, 10] < GRIP_CLOSED
        resid = float(np.median(grasp_err[gripped])) if gripped.any() else float("nan")
        slotted = first_slotted(d_seat)

        frames = []
        for k in range(len(S)):
            o = S[k]
            d.qpos[jadr] = o[11:18]
            d.qpos[f1] = d.qpos[f2] = o[10] / 2.0
            q = quat_mul(Q_L0_SEAT, o[3:7] / (np.linalg.norm(o[3:7]) + 1e-9))
            tip = SEAT + R_L0_SEAT @ o[0:3]
            d.qpos[bq:bq+3] = tip - axis_z_of(q) * TIP_OFF
            d.qpos[bq+3:bq+7] = q
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=cam)
            img = Image.fromarray(r.render())
            dr = ImageDraw.Draw(img)
            dr.text((8, 6), p.stem, fill=(200, 220, 255))
            dr.text((8, 22), f"step {k:4d}/{len(S)}  t {k/RATE:5.1f}s", fill=(255, 255, 80))
            dr.text((8, 38), f"d_seat {d_seat[k]*1000:6.1f} mm   tilt {tilt[k]:4.1f} deg", fill=(255, 255, 80))
            dr.text((8, 54), f"gripper {o[10]*1000:5.1f} mm" + ("  HOLDING" if o[10] < GRIP_CLOSED else "  released"),
                    fill=(255, 255, 80))
            if slotted is not None and k >= slotted:
                dr.text((8, 70), "SLOTTED (d_seat < 20 mm, held 1 s)", fill=(120, 255, 120))
            frames.append(np.asarray(img))
        imageio.mimsave(out.as_posix(), frames, fps=args.fps)
        print(f"{p.stem:<46} {len(S):>6} {d_seat[-1]*1000:7.1f}m {tilt[-1]:6.1f}d "
              f"{str(slotted):>7} {resid*1000:6.1f}m", flush=True)
        rows.append((len(S), d_seat[-1], tilt[-1], slotted, resid))
    r.close()
    if rows:
        n_ok = sum(1 for x in rows if x[3] is not None)
        print("-" * len(hdr))
        print(f"{n_ok}/{len(rows)} slotted; final d_seat "
              f"{np.min([x[1] for x in rows])*1000:.1f}-{np.max([x[1] for x in rows])*1000:.1f} mm, "
              f"tilt {np.min([x[2] for x in rows]):.1f}-{np.max([x[2] for x in rows]):.1f} deg, "
              f"median grasp residual {1000*np.median([x[4] for x in rows]):.1f} mm")
    print(f"videos in {args.outdir}")


if __name__ == "__main__":
    main()
