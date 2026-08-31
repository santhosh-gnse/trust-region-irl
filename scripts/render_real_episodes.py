"""Batch-replay REAL franka_bulbscrew episodes in the bulbscrew_mjx scene.

Real episodes carry no qpos, only the 25-D observation, so the pose is
reconstructed:

    arm joints  <- obs[11:18]        (joint encoders)
    fingers     <- obs[10] / 2 each
    bulb        <- tip = SEAT + R_L0_SEAT @ obs[0:3],  quat = Q_L0_SEAT * obs[3:7]
                   body origin = tip - axis_z * TIP_OFF

THE FRAME TRAP (bulbscrew_real_2026-08-30/SIM_CHANGES.md section 6): obs[0:3]
and obs[3:7] are in the SEAT frame, which is rotated 150.7 deg about z relative
to fr3_link0. Treating them as base-frame coordinates rotates the bulb about the
socket and puts it ~0.4 m from where it really is -- plausible-looking, with no
error raised anywhere.

The replay doubles as an alignment check: the arm comes from the robot's
encoders and the bulb from mocap, so when the jaws are closed the reconstructed
grasp point must coincide with the FK tool position (~8 mm on this rig; it was
~400 mm before the frame fix, and 1830 mm on the episode excluded for bad
tracking).

Usage:
    MUJOCO_GL=egl python scripts/render_real_episodes.py --data-dir <episodes/> \
        --outdir real_episode_videos
"""
import argparse
from pathlib import Path

import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "trust_region_irl" / "environments" / "bulbscrew_mjx" / "data" / "scene_mjx_bulb.xml"

TIP_OFF = -0.053                                  # canonical origin -> screw tip
SEAT = np.array([0.5993, 0.0972, 0.0672])         # measured 2026-08-30, fr3_link0
SEAT_OFF = np.array([0.0, 0.0, 0.018])            # socket body origin -> seat site
PLANK_TOP = 0.0389                                # measured with the closed gripper
# seat frame -> fr3_link0 (conj(base_q) * fixed_socket_quaternion_wxyz)
Q_L0_SEAT = np.array([0.253245, 0.001505, -0.005415, 0.967386])
R_L0_SEAT = np.array([[-0.871729, -0.489988,  0.000169],
                      [ 0.489955, -0.871675, -0.011239],
                      [ 0.005654, -0.009714,  0.999937]])
WS_LO = np.array([0.35, -0.35, 0.02])
WS_HI = np.array([0.85,  0.30, 0.45])
DEPTH_SOLVED = 0.003          # SIM_CHANGES section 2: screwed home, not just resting
HOLD_S = 1.0


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def axis_z_of(q):
    w, x, y, z = q
    return np.array([2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)])


def first_screwed(d_seat, rate=20, hold_s=HOLD_S, thr=DEPTH_SOLVED):
    """First step where d_seat < thr and stays under for hold_s (SIM_CHANGES 2)."""
    n = int(hold_s * rate)
    under = d_seat < thr
    for i in range(len(under) - n + 1):
        if under[i:i + n].all():
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=REPO / "real_episode_videos")
    ap.add_argument("--fps", type=int, default=40)      # 2x real time
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip episodes whose mp4 is already present (resumable)")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(MODEL.as_posix())
    m.vis.global_.offwidth, m.vis.global_.offheight = args.width, args.height
    d = mujoco.MjData(m)

    # No runtime patching needed: the scene now carries the measured seat, plank
    # height and free-standing bulb spawn (SIM_CHANGES.md applied). Verify rather
    # than assume, so a future scene edit cannot silently move the replay.
    seat_in_model = m.body_pos[m.body("socket").id] + SEAT_OFF
    if not np.allclose(seat_in_model, SEAT, atol=1e-3):
        raise SystemExit(f"scene seat {seat_in_model} != measured {SEAT}; "
                         "re-apply SIM_CHANGES.md section 3 before rendering")

    jadr = [m.joint(f"fr3_joint{i+1}").qposadr[0] for i in range(7)]
    f1 = m.joint("finger_joint1").qposadr[0]
    f2 = m.joint("finger_joint2").qposadr[0]
    bq = m.joint("bulb_joint").qposadr[0]
    ee_bid = m.body("ee_frame").id

    r = mujoco.Renderer(m, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.64, 0.03, 0.10]
    cam.distance, cam.elevation, cam.azimuth = 0.95, -22, 150

    args.outdir.mkdir(exist_ok=True)
    episodes = sorted(args.data_dir.glob("*.npz"))
    print(f"{len(episodes)} episodes -> {args.outdir}\n")
    hdr = f"{'episode':<52} {'steps':>6} {'best':>7} {'final':>7} {'screwed':>8} {'grasp':>7}"
    print(hdr); print("-" * len(hdr))

    summary = []
    for path in episodes:
        out = args.outdir / f"{path.stem}.mp4"
        if args.skip_existing and out.exists():
            print(f"{path.stem:<52} {'':>6} {'':>7} {'':>7} {'skip':>8}", flush=True)
            continue
        S = np.load(path)["states"]
        d_seat = np.linalg.norm(S[:, 0:3], axis=1)
        grasp_err = np.linalg.norm(S[:, 7:10], axis=1)
        gripped = S[:, 10] < 0.065
        resid = float(np.median(grasp_err[gripped])) if gripped.any() else float("nan")
        screwed = first_screwed(d_seat)

        frames = []
        for k in range(0, len(S), args.stride):
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
            dr.text((8, 6), f"{path.stem}", fill=(200, 220, 255))
            dr.text((8, 22), f"step {k:4d}/{len(S)}  t {k/20:5.1f}s", fill=(255, 255, 80))
            dr.text((8, 38), f"d_seat {d_seat[k]*1000:6.1f} mm", fill=(255, 255, 80))
            dr.text((8, 54), f"gripper {o[10]*1000:5.1f} mm   tool-grasp {grasp_err[k]*1000:5.1f} mm",
                    fill=(255, 255, 80))
            tcp = d.xpos[ee_bid]
            if np.any(tcp < WS_LO) or np.any(tcp > WS_HI):
                dr.text((args.width - 230, 6), "TOOL OUTSIDE WORKSPACE", fill=(255, 120, 120))
            if screwed is not None and k >= screwed:
                dr.text((8, 70), "SCREWED HOME (d_seat < 3 mm, held 1 s)", fill=(120, 255, 120))
            frames.append(np.asarray(img))

        imageio.mimsave(out.as_posix(), frames, fps=args.fps)
        print(f"{path.stem:<52} {len(S):>6} {d_seat.min()*1000:6.1f}m {d_seat[-1]*1000:6.1f}m "
              f"{str(screwed):>8} {resid*1000:6.1f}m", flush=True)
        summary.append((path.stem, len(S), d_seat.min(), d_seat[-1], screwed, resid))
    r.close()

    print("-" * len(hdr))
    ok = [s for s in summary if s[4] is not None]
    print(f"{len(ok)}/{len(summary)} episodes reach 'screwed home'; "
          f"median tool-grasp residual {1000*np.median([s[5] for s in summary]):.1f} mm")
    print(f"videos in {args.outdir}")


if __name__ == "__main__":
    main()
