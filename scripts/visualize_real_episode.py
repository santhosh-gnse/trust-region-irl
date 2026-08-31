"""Replay a REAL kinesthetic episode (franka_bulbscrew) in the bulbscrew_mjx scene.

Real episodes carry no qpos -- only the 25-D observation -- so the pose is
reconstructed from it:

    arm joints  <- obs[11:18]                    (measured directly)
    fingers     <- obs[10] / 2 each
    bulb        <- tip = seat + obs[0:3], quat = obs[3:7]; the body origin is
                   then tip - axis_z * TIP_OFF

This doubles as an alignment check: the arm comes from joint encoders and the
bulb from mocap, so if the calibration is right the bulb should sit between the
fingers whenever the demonstration has it grasped.

Usage:
    MUJOCO_GL=egl python scripts/visualize_real_episode.py --data ../episode_0000.npz
"""
import argparse
from pathlib import Path

import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "trust_region_irl" / "environments" / "bulbscrew_mjx" / "data" / "scene_mjx_bulb.xml"
TIP_OFF = -0.053          # bulbscrew_mjx BulbScrew.TIP_OFF
SEAT_OFF = np.array([0.0, 0.0, 0.018])


def axis_z_of(q):
    w, x, y, z = q
    return np.array([2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "bulb_expert_videos" / "real_episode_0000.mp4")
    ap.add_argument("--fps", type=int, default=40)      # 2x real time
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-reach", type=float, default=0.6,
                    help="bulb readings farther than this from the seat are treated as "
                         "mocap dropouts: the last good pose is held and the frame flagged")
    args = ap.parse_args()

    d = np.load(args.data)
    S = d["states"]
    m = mujoco.MjModel.from_xml_path(MODEL.as_posix())
    m.vis.global_.offwidth, m.vis.global_.offheight = 960, 720
    data = mujoco.MjData(m)

    jadr = [m.joint(f"fr3_joint{i+1}").qposadr[0] for i in range(7)]
    f1 = m.joint("finger_joint1").qposadr[0]
    f2 = m.joint("finger_joint2").qposadr[0]
    bq = m.joint("bulb_joint").qposadr[0]
    seat = data.xpos[m.body("socket").id].copy()
    mujoco.mj_forward(m, data)
    seat = data.xpos[m.body("socket").id] + SEAT_OFF

    # success criterion (bulbscrew_mjx / extract_success_trajectories)
    quat = np.where(S[:, 3:4] < 0, -S[:, 3:7], S[:, 3:7])
    d_seat = np.linalg.norm(S[:, 0:3], axis=1)
    orn = 2*np.arccos(np.clip(quat[:, 0], -1.0, 1.0))
    task_err = d_seat + 0.1*orn
    solved_at = np.flatnonzero(task_err < 0.02)
    solved_at = int(solved_at[0]) if solved_at.size else None

    # Mocap dropouts: the bulb is sometimes tracked ~1.85 m from the seat --
    # behind the robot, i.e. Motive locked onto something that is not the bulb.
    # Hold the last plausible pose through those runs so the demonstration is
    # watchable, and flag every affected frame rather than hiding it.
    dropout = d_seat > args.max_reach
    print(f"mocap dropouts: {dropout.sum()}/{len(S)} frames ({100*dropout.mean():.1f}%)")

    r = mujoco.Renderer(m, height=720, width=960)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.67, -0.03, 0.10]; cam.distance = 0.95
    cam.elevation = -22; cam.azimuth = 150

    frames = []
    for k in range(0, len(S), args.stride):
        o = S[k]
        data.qpos[jadr] = o[11:18]
        data.qpos[f1] = data.qpos[f2] = o[10] / 2.0
        if dropout[k]:
            src = np.flatnonzero(~dropout[:k + 1])
            ref = S[src[-1]] if src.size else o
        else:
            ref = o
        q = ref[3:7] / (np.linalg.norm(ref[3:7]) + 1e-9)
        tip = seat + ref[0:3]
        data.qpos[bq:bq+3] = tip - axis_z_of(q) * TIP_OFF
        data.qpos[bq+3:bq+7] = q
        mujoco.mj_forward(m, data)
        r.update_scene(data, camera=cam)
        img = Image.fromarray(r.render())
        dr = ImageDraw.Draw(img)
        tag = "SOLVED" if (solved_at is not None and k >= solved_at) else ""
        dr.text((10, 8), f"step {k:4d}/{len(S)}   t {k/20:5.1f}s", fill=(255, 255, 80))
        dr.text((10, 24), f"task_err {task_err[k]:.4f}   d_seat {d_seat[k]:.4f}", fill=(255, 255, 80))
        dr.text((10, 40), f"gripper {o[10]*1000:5.1f} mm", fill=(255, 255, 80))
        if tag:
            dr.text((10, 56), tag, fill=(120, 255, 120))
        if dropout[k]:
            dr.text((760, 8), "BULB TRACKING LOST", fill=(255, 120, 120))
        frames.append(np.asarray(img))
    r.close()

    args.out.parent.mkdir(exist_ok=True)
    imageio.mimsave(args.out.as_posix(), frames, fps=args.fps)
    print(f"solved first at step {solved_at} (task_err<0.02); min {task_err.min():.4f} @ {task_err.argmin()}")
    print(f"wrote {args.out} ({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
