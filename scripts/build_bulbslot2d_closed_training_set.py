"""Assemble the bulbslot2d_closed_mjx training set: same real demonstrations
and same plane-projected relabeling as bulbslot2d_mjx, but with the gripper
action column DROPPED (2-D actions only) and truncated at DEPTH ALONE, not
depth+release -- this task version never releases, so the natural endpoint is
where the original bulbslot_mjx also ends: d_seat < 20mm held 1s.

Reuses trirl_dataset/real_expert/slot2d_relabelled/*.npz (the full, untruncated,
plane-relabelled per-episode files already built for bulbslot2d_mjx) rather
than re-deriving anything -- states and the linear (u, z) action columns are
identical; only the truncation point and the dropped 3rd action column differ.

Usage:
    python scripts/build_bulbslot2d_closed_training_set.py
"""
import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DT = 0.05
DEPTH_SOLVED, HOLD_S = 0.020, 1.0


def first_solved(d_seat, thr=DEPTH_SOLVED, hold=HOLD_S, dt=DT):
    n = int(round(hold / dt))
    under = d_seat < thr
    for i in range(len(under) - n + 1):
        if under[i:i + n].all():
            return i + n - 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path,
                    default=REPO / "trirl_dataset" / "real_expert" / "slot2d_relabelled")
    ap.add_argument("--out", type=Path,
                    default=REPO / "trirl_dataset" / "rl_expert" / "expert_dataset_bulbslot2d_closed_real.npz")
    args = ap.parse_args()

    eps = sorted(args.in_dir.glob("*.npz"))
    S, A, NS, AB, R, lens, dropped = [], [], [], [], [], [], []
    print(f"{len(eps)} episodes")
    print(f"{'episode':<46} {'full':>5} {'kept':>5} {'cut@':>5}")
    print("-" * 66)
    for p in eps:
        z = np.load(p)
        st, ac, ns = z["states"], z["actions"], z["next_states"]
        d_seat = np.linalg.norm(st[:, 0:3], axis=1)
        cut = first_solved(d_seat)
        if cut is None:
            dropped.append(p.stem)
            print(f"{p.stem:<46} {len(st):>5} {'DROP':>5} {'never':>5}")
            continue
        n = cut + 1
        ab = np.zeros(n, np.float32); ab[-1] = 1.0
        S.append(st[:n]); A.append(ac[:n, :2].astype(np.float32)); NS.append(ns[:n])
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
    print("-" * 66)
    print(f"{len(lens)} episodes, {lens.sum()} transitions ({lens.sum()*DT:.0f} s at 20 Hz)")
    print(f"lengths: min {lens.min()} median {int(np.median(lens))} max {lens.max()}")
    print(f"kept {100*lens.sum()/tot:.0f}% of the recorded transitions; {len(dropped)} dropped")
    print(f"action shape: {A[0].shape[1]}-D (u, z only -- gripper column dropped)")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
