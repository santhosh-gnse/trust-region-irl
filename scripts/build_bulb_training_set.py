"""Assemble the real bulb-screwing demonstrations into one TRIRL training npz.

Two things happen here:

  * TRUNCATION at success. Recorded episodes continue past the point where the
    bulb is screwed home -- the operator keeps adjusting, or does not press Stop
    immediately. Left in, that tail teaches a policy to keep fiddling with a bulb
    that is already seated. Each episode is cut at the first step where
    `d_seat < 3 mm` and stays under for 1.0 s, which is exactly where a
    bulbscrew_mjx episode terminates.

  * CONCATENATION. TRIRL's prepare_expert_data loads a single npz holding the
    transitions of every episode end to end, with `absorbing = 1.0` marking each
    episode's final step (dtype float32 -- a bool array trips a cast bug).

Usage:
    python scripts/build_bulb_training_set.py \
        --in-dir trirl_dataset/real_expert/relabelled_v1 \
        --out    trirl_dataset/rl_expert/expert_dataset_bulbscrew_real_27.npz
"""
import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DT = 0.05
DEPTH, HOLD_S = 0.003, 1.0          # bulbscrew_mjx DEPTH_SOLVED / HOLD_SECONDS


def first_screwed(d_seat, thr=DEPTH, hold=HOLD_S, dt=DT):
    n = int(round(hold / dt))
    under = d_seat < thr
    for i in range(len(under) - n + 1):
        if under[i:i + n].all():
            return i + n - 1        # cut at the END of the hold: it is solved there
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-truncate", action="store_true",
                    help="keep full episodes (post-success tails included)")
    args = ap.parse_args()

    eps = sorted(args.in_dir.glob("*.npz"))
    S, A, NS, AB, R = [], [], [], [], []
    lens, dropped = [], []
    print(f"{'episode':<50} {'full':>6} {'kept':>6} {'cut@':>6}")
    print("-" * 72)
    for p in eps:
        z = np.load(p)
        d_seat = np.linalg.norm(z["states"][:, 0:3], axis=1)
        cut = None if args.no_truncate else first_screwed(d_seat)
        if not args.no_truncate and cut is None:
            dropped.append(p.stem)
            print(f"{p.stem:<50} {len(d_seat):>6} {'DROP':>6} {'never':>6}")
            continue
        n = len(d_seat) if cut is None else cut + 1
        ab = np.zeros(n, np.float32)
        ab[-1] = 1.0                                  # terminal step of this episode
        S.append(z["states"][:n]); A.append(z["actions"][:n])
        NS.append(z["next_states"][:n]); AB.append(ab)
        R.append(np.zeros(n, np.float32))             # TRIRL learns its own reward
        lens.append(n)
        print(f"{p.stem:<50} {len(d_seat):>6} {n:>6} {str(cut):>6}")

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
    print("-" * 72)
    print(f"{len(lens)} episodes, {lens.sum()} transitions ({lens.sum()*DT:.0f} s at 20 Hz)")
    print(f"lengths: min {lens.min()} median {int(np.median(lens))} max {lens.max()} "
          f"(sim horizon 1000)")
    kept = 100 * lens.sum() / sum(len(np.load(p)['states']) for p in eps)
    print(f"kept {kept:.0f}% of the recorded transitions; {len(dropped)} episode(s) dropped")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
