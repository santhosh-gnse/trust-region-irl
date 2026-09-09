"""Replay real pizza-spreading episodes in the pizza_mjx scene.

Real episodes carry no qpos, but they carry `arm_qpos` (obs[3:10]) straight from
the joint encoders, so the arm pose is exact -- no frame reconstruction is
needed here, unlike the bulb datasets whose object poses came from mocap in a
rotated seat frame.

Each frame also carries a top-down COVERAGE panel: the pizza disc with the cells
the tool has swept so far filled in. That is what makes spreading legible -- the
3-D view alone shows a tool moving, not a surface being covered. The same polar
grid the environment uses (8 radial x 24 angular over a 156 mm radius) is used
here, so the panel and the env's coverage channel agree by construction.

Usage:
    MUJOCO_GL=egl python scripts/render_pizza_episodes.py --data-dir <dir> --outdir pizza_videos
"""
import argparse
from pathlib import Path

import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "trust_region_irl" / "environments" / "pizza_mjx" / "data" / "scene_mjx_pizza.xml"
# geometry comes from the environment so the two can never drift apart
from trust_region_irl.environments.pizza_mjx.environment import PizzaSpread as _P
PIZZA_OFF, PIZZA_R, TOOL_R = _P.PIZZA_OFF, _P.PIZZA_R, _P.TOOL_R
N_RAD, N_ANG = _P.N_RAD, _P.N_ANG
RATE = 20


_IR = (np.arange(N_RAD) + 0.5) / N_RAD * PIZZA_R
_IA = (np.arange(N_ANG) + 0.5) / N_ANG * 2 * np.pi - np.pi
CELL_C = np.stack([np.outer(_IR, np.cos(_IA)), np.outer(_IR, np.sin(_IA))], -1).reshape(-1, 2)


def mark(visited, xy):
    """Mark every cell the 30 mm tool DISC touches, not just the one under its
    centre -- the same footprint model the environment uses."""
    q = np.asarray(xy) - PIZZA_OFF
    r = float(np.linalg.norm(q))
    if r <= PIZZA_R:
        ir = min(int(r / PIZZA_R * N_RAD), N_RAD - 1)
        ia = min(int((float(np.arctan2(q[1], q[0])) + np.pi) / (2 * np.pi) * N_ANG), N_ANG - 1)
        visited[ir * N_ANG + ia] = True
    visited |= np.linalg.norm(CELL_C - q, axis=1) <= TOOL_R
    return visited


def coverage_panel(visited, path_xy, size=210):
    """Top-down pizza with swept cells filled and the tool path drawn."""
    img = Image.new("RGB", (size, size), (18, 22, 30))
    d = ImageDraw.Draw(img)
    c = size // 2
    scale = (size * 0.44) / PIZZA_R
    d.ellipse([c - PIZZA_R * scale, c - PIZZA_R * scale,
               c + PIZZA_R * scale, c + PIZZA_R * scale], fill=(70, 55, 42))
    for idx in np.flatnonzero(visited):
        ir, ia = divmod(int(idx), N_ANG)
        r0, r1 = ir / N_RAD * PIZZA_R, (ir + 1) / N_RAD * PIZZA_R
        # PIL measures angles clockwise in a y-DOWN image, while the cell index
        # and the drawn path are both in y-up maths coords -- negate, or the
        # filled wedges come out mirrored about the x-axis from the tool.
        a1 = -(ia / N_ANG * 360.0 - 180.0)
        a0 = -((ia + 1) / N_ANG * 360.0 - 180.0)
        d.pieslice([c - r1 * scale, c - r1 * scale, c + r1 * scale, c + r1 * scale],
                   a0, a1, fill=(150, 34, 22))
        if r0 > 0:
            d.pieslice([c - r0 * scale, c - r0 * scale, c + r0 * scale, c + r0 * scale],
                       a0, a1, fill=(70, 55, 42))
    d.ellipse([c - PIZZA_R * scale, c - PIZZA_R * scale,
               c + PIZZA_R * scale, c + PIZZA_R * scale], outline=(230, 220, 200), width=2)
    if len(path_xy) > 1:
        pts = [(c + (p[0] - PIZZA_OFF[0]) * scale,
                c - (p[1] - PIZZA_OFF[1]) * scale) for p in path_xy]
        d.line(pts, fill=(255, 235, 120), width=1)
        d.ellipse([pts[-1][0] - 3, pts[-1][1] - 3, pts[-1][0] + 3, pts[-1][1] + 3],
                  fill=(255, 255, 255))
    d.text((6, 4), f"coverage {100*visited.mean():5.1f}%", fill=(240, 240, 240))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=REPO / "pizza_episode_videos")
    ap.add_argument("--fps", type=int, default=40)          # 2x real time
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=608)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(MODEL.as_posix())
    m.vis.global_.offwidth, m.vis.global_.offheight = args.width, args.height
    d = mujoco.MjData(m)
    jadr = [m.joint(f"fr3_joint{i+1}").qposadr[0] for i in range(7)]

    r = mujoco.Renderer(m, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.6097, -0.0069, 0.05]
    cam.distance, cam.elevation, cam.azimuth = 0.95, -38, 140

    args.outdir.mkdir(exist_ok=True)
    eps = sorted(args.data_dir.rglob("*.npz"))
    print(f"{len(eps)} episodes -> {args.outdir}\n")
    hdr = f"{'episode':<46} {'steps':>6} {'coverage':>9} {'max r':>8} {'|dz|max':>8}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for p in eps:
        name = f"{p.parent.name}__{p.stem}"
        out = args.outdir / f"{name}.mp4"
        if args.skip_existing and out.exists():
            continue
        S = np.load(p)["states"]
        visited = np.zeros(N_RAD * N_ANG, dtype=bool)
        path, frames = [], []
        for k in range(len(S)):
            d.qpos[jadr] = S[k, 3:10]
            mujoco.mj_forward(m, d)
            xy = S[k, 0:2]
            path.append(xy.copy())
            visited = mark(visited, xy)
            r.update_scene(d, camera=cam)
            img = Image.fromarray(r.render())
            img.paste(coverage_panel(visited, path), (args.width - 216, 6))
            dr = ImageDraw.Draw(img)
            dr.text((8, 6), name, fill=(200, 220, 255))
            dr.text((8, 22), f"step {k:4d}/{len(S)}   t {k/RATE:5.1f}s", fill=(255, 255, 80))
            spd = (np.linalg.norm(S[min(k+1, len(S)-1), 0:2] - xy) * RATE
                   if len(S) > 1 else 0.0)
            dr.text((8, 38), f"radius {np.linalg.norm(xy - PIZZA_OFF)*1000:6.1f} mm   "
                             f"height {S[k,2]*1000:+6.1f} mm", fill=(255, 255, 80))
            dr.text((8, 54), f"speed  {spd*1000:6.1f} mm/s", fill=(120, 255, 160))
            frames.append(np.asarray(img))
        imageio.mimsave(out.as_posix(), frames, fps=args.fps)
        rr = np.linalg.norm(S[:, :2] - PIZZA_OFF, axis=1)
        print(f"{name:<46} {len(S):>6} {100*visited.mean():8.1f}% {rr.max()*1000:6.1f}mm "
              f"{np.abs(S[:,2]).max()*1000:6.1f}mm", flush=True)
        rows.append((len(S), visited.mean(), rr.max()))
    r.close()
    if rows:
        cov = np.array([x[1] for x in rows])
        print("-" * len(hdr))
        print(f"{len(rows)} episodes | coverage {100*cov.min():.0f}-{100*cov.max():.0f}% "
              f"(median {100*np.median(cov):.0f}%) | max radius up to "
              f"{max(x[2] for x in rows)*1000:.0f} mm")
    print(f"videos in {args.outdir}")


if __name__ == "__main__":
    main()
