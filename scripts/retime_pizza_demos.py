"""Re-time pizza demonstrations onto a canonical speed profile (+ tempo jitter).

WHY. Hand-guided demos are geometrically good but temporally messy: 34% of every
episode is near-stationary dwell, within-episode speed sd is 107% of the mean,
and episode duration varies 2.9x while the path length varies only 2.4x -- so
most of the timing spread is pauses, not different paths. MCE-IRL matches feature
EXPECTATIONS, which are time-averages, so dwell silently reweights whichever
radius the demonstrator happened to pause at. Re-timing fixes the target the
reward is fit to without touching the path that was actually demonstrated.

WHAT IS PRESERVED. Speed here is not noise: the demonstrator moves slowly at the
centre (23 mm/s), fastest at mid-radius (350 mm/s) and slows again at the rim
(180 mm/s). That profile is fitted once from the pooled data and imposed on every
episode, so the deliberate structure survives while the per-episode tempo noise
and the pauses do not.

HOW. Re-parameterise each path by arc length (which drops dwell automatically,
since a pause advances no arc length), integrate dt = ds / v*(r) to get a new
time base, resample the joint trajectory on it, then re-derive velocities and
actions by forward differencing exactly as the v2 relabeller does -- so every
transition stays dynamically consistent.

TEMPO JITTER. Optionally emit several copies per episode at global tempo factors.
Re-timing alone makes the data one specific tempo; jitter makes the learned
reward tempo-INVARIANT rather than tempo-blind by luck.
"""
import argparse
from pathlib import Path

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "trust_region_irl" / "environments" / "pizza_mjx" / "data" / "scene_mjx_pizza.xml"
from trust_region_irl.environments.pizza_mjx.environment import PizzaSpread as _P
PIZZA_OFF, PIZZA_R = _P.PIZZA_OFF, _P.PIZZA_R


def coverage_track(xy, off, R, tool_r, n_rad, n_ang):
    """Running coverage fraction after marking each xy in sequence -- the same
    footprint-disc accumulator PizzaSpread._mark uses, replayed offline so the
    expert dataset carries the identical channel the env's obs_mode="coverage"
    produces. A per-transition feature cannot recover "been swept" from position
    alone, so the discriminator needs it pre-computed and baked into the state."""
    ir = (np.arange(n_rad) + 0.5) / n_rad * R
    ia = (np.arange(n_ang) + 0.5) / n_ang * 2 * np.pi - np.pi
    cc = np.stack([np.outer(ir, np.cos(ia)), np.outer(ir, np.sin(ia))], -1).reshape(-1, 2)
    visited = np.zeros(n_rad * n_ang, dtype=bool)
    cov = np.empty(len(xy), dtype=np.float32)
    for k, p in enumerate(xy):
        q = p - off
        r = np.linalg.norm(q)
        if r <= R:
            th = np.arctan2(q[1], q[0])
            iir = min(int(r / R * n_rad), n_rad - 1)
            iia = min(int((th + np.pi) / (2 * np.pi) * n_ang), n_ang - 1)
            visited[iir * n_ang + iia] = True
        visited |= np.linalg.norm(cc - q, axis=1) <= tool_r
        cov[k] = visited.mean()
    return cov
RATE = 20.0
DT = 1.0 / RATE


def fwd_diff(x, dt):
    """Forward difference; the v2 convention -- a_t must be the action that moves
    x_t to x_{t+1}, which centred differencing would smear across both."""
    v = np.diff(x, axis=0) / dt
    return np.concatenate([v, v[-1:]], axis=0)


def speed_profile(eps, fk_xy, nbin=10, smooth=2, moving=0.005):
    """Median tool speed as a function of radius, pooled over all episodes.

    Only MOVING samples count. Dwell is concentrated near the centre, where it is
    a third of all samples, so a median over everything reads 0 mm/s there and
    the profile would ask the tool to stop at the start of every spiral. What we
    want is how fast the demonstrator travels when travelling."""
    R, V = [], []
    for e in eps:
        xy = fk_xy[e]
        V.append(np.linalg.norm(np.diff(xy, axis=0), axis=1) / DT)
        R.append(np.linalg.norm(xy[:-1] - PIZZA_OFF, axis=1))
    R, V = np.concatenate(R), np.concatenate(V)
    mv = V > moving
    R, V = R[mv], V[mv]
    edges = np.linspace(0.0, PIZZA_R, nbin + 1)
    prof = np.array([np.median(V[(R >= edges[i]) & (R < edges[i + 1])])
                     if ((R >= edges[i]) & (R < edges[i + 1])).any() else np.nan
                     for i in range(nbin)])
    prof = np.interp(np.arange(nbin), np.flatnonzero(~np.isnan(prof)), prof[~np.isnan(prof)])
    if smooth > 1:                       # moving average, edges held
        k = np.ones(smooth) / smooth
        prof = np.convolve(np.pad(prof, (smooth // 2, smooth // 2), mode="edge"), k, "same")
        prof = prof[smooth // 2: smooth // 2 + nbin]
    return 0.5 * (edges[:-1] + edges[1:]), prof


def retime(S, xy, centres, prof, v_peak, tempo, floor=0.02):
    """Resample one episode onto the canonical profile scaled to v_peak*tempo.

    `xy` must be the FK tool path of S's joint angles, not the logged EE pose.
    The output is rebuilt by FK from resampled joint angles, so parameterising by
    the logged path instead leaves the achieved speed off target wherever the two
    disagree -- which is what pushed p90 to 366 mm/s against a 354 mm/s cap."""
    q = S[:, 3:10]
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(step)])
    keep = np.concatenate([[True], np.diff(s) > 1e-6])     # drop dwell samples
    s, q, xy = s[keep], q[keep], xy[keep]
    if len(s) < 4:
        return None

    # target speed along the path, from the radius at each arc-length sample
    v_of = prof / prof.max() * v_peak * tempo
    r = np.linalg.norm(xy - PIZZA_OFF, axis=1)
    v = np.maximum(np.interp(r, centres, v_of), floor)

    # t(s) by trapezoidal integration of ds / v
    dt_seg = np.diff(s) / (0.5 * (v[:-1] + v[1:]))
    t = np.concatenate([[0.0], np.cumsum(dt_seg)])
    n = max(int(np.floor(t[-1] / DT)) + 1, 4)
    t_new = np.arange(n) * DT
    s_new = np.interp(t_new, t, s)
    return np.stack([np.interp(s_new, s, q[:, j]) for j in range(7)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to expert_dataset_pizza_real[_coverage].npz under "
                         "trirl_dataset/rl_expert/, named by --obs-mode")
    ap.add_argument("--obs-mode", choices=["plain", "coverage"], default="plain",
                    help="plain (17-D, matches the raw sensor data exactly) or "
                         "coverage (18-D, appends the running swept-coverage fraction "
                         "-- required by feature_fn=spread / obs_mode=coverage envs)")
    ap.add_argument("--v-peak", type=float, default=0.10,
                    help="peak tool speed of the canonical profile (m/s)")
    ap.add_argument("--tempos", type=float, nargs="+", default=[1.0],
                    help="global tempo factors; >1 emits jitter-augmented copies")
    ap.add_argument("--max-speed", type=float, default=None,
                    help="action scale; defaults to v_peak * fastest tempo, since "
                         "a lower value would clip the jitter-augmented copies")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if args.out is None:
        suffix = "_coverage" if args.obs_mode == "coverage" else ""
        args.out = (REPO / "trirl_dataset" / "rl_expert" /
                   f"expert_dataset_pizza_real{suffix}.npz")
    if args.max_speed is None:
        args.max_speed = args.v_peak * max(args.tempos)
        print(f"action scale (max_speed) set to {args.max_speed:.3f} m/s "
              f"= v_peak x fastest tempo\n")
    eps = sorted(args.data_dir.rglob("*.npz"))
    m = mujoco.MjModel.from_xml_path(MODEL.as_posix())
    d = mujoco.MjData(m)
    jadr = [m.joint(f"fr3_joint{i+1}").qposadr[0] for i in range(7)]
    eeid = m.body("ee_frame").id
    d.qpos[jadr] = np.load(eps[0])["states"][0, 3:10]
    mujoco.mj_forward(m, d)

    def fk(Q):
        P = np.empty((len(Q), 3))
        for i, qq in enumerate(Q):
            d.qpos[jadr] = qq
            mujoco.mj_kinematics(m, d)
            P[i] = d.xpos[eeid]
        return P

    home = np.array([0.6247, -0.0094, 0.0641])
    fk_xy = {e: (fk(np.load(e)["states"][:, 3:10]) - home)[:, :2] for e in eps}
    centres, prof = speed_profile(eps, fk_xy)
    print("canonical speed profile (median over all demos, scaled to v_peak):")
    for c, v in zip(centres, prof / prof.max() * args.v_peak):
        print(f"   r {1000*c:5.1f} mm -> {1000*v:6.1f} mm/s")

    out_states, out_actions, out_next, out_abs, out_lengths, stats = [], [], [], [], [], []
    for e in eps:
        S = np.load(e)["states"]
        xy_fk = fk_xy[e]
        for tempo in args.tempos:
            Q = retime(S, xy_fk, centres, prof, args.v_peak, tempo)
            if Q is None:
                continue
            P = fk(Q) - home
            qd = fwd_diff(Q, DT)
            vxy = fwd_diff(P[:, :2], DT)
            a = np.clip(vxy / args.max_speed, -1.0, 1.0)
            st = np.concatenate([P, Q, qd], axis=1).astype(np.float32)
            if args.obs_mode == "coverage":
                cov = coverage_track(P[:, :2], _P.PIZZA_OFF, _P.PIZZA_R, _P.TOOL_R,
                                     _P.N_RAD, _P.N_ANG)
                st = np.concatenate([st, cov[:, None]], axis=1)
            out_states.append(st)
            out_actions.append(a.astype(np.float32))
            # next_states: shift by one step; absorbing marks each episode's last
            # transition, matching the bulbslot/bulbscrew dataset convention that
            # prepare_expert_data (trust_region_irl/algorithms/data_utils.py) expects
            out_next.append(np.concatenate([st[1:], st[-1:]], axis=0))
            ab = np.zeros(len(st), dtype=np.float32); ab[-1] = 1.0
            out_abs.append(ab)
            out_lengths.append(len(st))
            sp = np.linalg.norm(vxy, axis=1)
            # real clipping = the commanded SPEED had to be cut, not merely an
            # axis-aligned move touching the box boundary
            stats.append((len(st), sp.mean(), sp.std(),
                          float((sp > args.max_speed * 1.001).mean()),
                          float((sp < 0.005).mean())))
    stats = np.array(stats)
    print(f"\n{len(out_states)} episodes ({len(eps)} demos x {len(args.tempos)} tempos)")
    print(f"  steps       {stats[:,0].mean():7.1f} +- {stats[:,0].std():.1f}"
          f"   ({stats[:,0].min():.0f} .. {stats[:,0].max():.0f})")
    print(f"  tool speed  {stats[:,1].mean():7.4f} m/s, within-episode sd/mean "
          f"{(stats[:,2]/stats[:,1]).mean():.2f}   (raw data: 1.07)")
    print(f"  action clipped {100*stats[:,3].mean():.1f}% of steps   (raw data: 75.6%)")
    print(f"  near-stationary {100*stats[:,4].mean():.1f}% of steps  (raw data: 34.1%)")
    if args.report_only:
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        states=np.concatenate(out_states), actions=np.concatenate(out_actions),
        next_states=np.concatenate(out_next),
        absorbing=np.concatenate(out_abs),
        rewards=np.zeros(sum(out_lengths), dtype=np.float32),  # reward is learned, not logged
        episode_lengths=np.array(out_lengths, dtype=np.int32),
        v_peak=args.v_peak, tempos=np.array(args.tempos), max_speed=args.max_speed)
    print(f"\nwrote {args.out}  ({sum(out_lengths)} transitions)")


if __name__ == "__main__":
    main()
