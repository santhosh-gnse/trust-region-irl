"""Replay bulbscrew FSM expert episodes from their exact qpos and write mp4s.

Usage (trirl env):
    MUJOCO_GL=egl python scripts/visualize_bulb_expert.py --episodes 0 1
"""
import argparse
from pathlib import Path

import numpy as np
import mujoco
import imageio.v2 as imageio

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO / "trirl_dataset" / "rl_expert" / "expert_dataset_bulbscrew_fsm.npz"
MODEL = REPO / "trust_region_irl" / "environments" / "bulbscrew_mjx" / "data" / "scene_mjx_bulb.xml"


def split_episodes(absorbing):
    ends = np.where(absorbing > 0.5)[0]
    ranges, start = [], 0
    for e in ends:
        ranges.append((start, e + 1))
        start = e + 1
    return ranges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--episodes", type=int, nargs="*", default=[0])
    ap.add_argument("--outdir", type=Path, default=REPO / "bulb_expert_videos")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    d = np.load(args.data)
    qpos, absorbing = d["qpos"], d["absorbing"]
    width = d["states"][:, 10]   # gripper opening (m)
    episodes = split_episodes(absorbing)
    print(f"loaded {len(qpos)} transitions, {len(episodes)} episodes")

    m = mujoco.MjModel.from_xml_path(MODEL.as_posix())
    m.vis.map.force = 0.05          # scale contact-force arrows to fingertip-force range
    data = mujoco.MjData(m)
    r = mujoco.Renderer(m, height=480, width=640)
    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.52, 0, 0.12]; cam.distance = 0.95; cam.elevation = -20; cam.azimuth = 155
    args.outdir.mkdir(exist_ok=True)

    for ep in args.episodes:
        s, e = episodes[ep]
        frames = []
        from PIL import Image, ImageDraw
        for k, q in enumerate(qpos[s:e]):
            data.qpos[:] = q
            mujoco.mj_forward(m, data)
            r.update_scene(data, camera=cam, scene_option=vopt)
            img = Image.fromarray(r.render())
            # 25-D observations carry no tactile (real hand has none); show the
            # gripper opening instead, which is what confirms a grasp here.
            ImageDraw.Draw(img).text((10, 8), f"gripper width {width[s + k]*1000:5.1f} mm",
                                     fill=(255, 255, 80))
            frames.append(np.asarray(img))
        out = args.outdir / f"bulb_expert_ep{ep:03d}.mp4"
        imageio.mimsave(out.as_posix(), frames, fps=args.fps)
        print(f"  wrote {out} ({len(frames)} frames)")
    r.close()


if __name__ == "__main__":
    main()
