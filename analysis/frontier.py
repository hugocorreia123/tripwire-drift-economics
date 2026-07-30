#!/usr/bin/env python3
r"""Best achievable schedule vs best achievable detector.

The single-point comparison in run_comparison.py assumed the schedule's
period k was fixed. On ELEC2 that assumption broke: B scored 0.8064 at
k=5 (8 retrains) and 0.7883 at k=4 (11 retrains) — MORE retraining
scored WORSE, because a short memory window means fewer rows per fit.
Performance is therefore not monotonic in retrain count, so "matched on
count" does not guarantee a fair comparison, and a detector can win
merely by landing on a luckier point of a bumpy curve.

A practitioner would tune whichever policy they adopted. So the honest
question is not "does this detector beat this schedule" but:

    does the best achievable detector beat the best achievable
    schedule, at equal cost?

This sweeps both and prints the frontier: performance against number of
retrains, for each policy. If the curves lie on top of each other, the
signal is worth nothing at any operating point — a far stronger claim
than a single tie.

    python3 analysis/frontier.py data/elec2.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.policies import (run_policy, count_fires, majority_baseline,
                               DETECTORS)
from analysis.real_stream import load_windows


def threshold_for_rate(windows, detector, n_target):
    """Detector threshold whose firing COUNT is closest to n_target."""
    lo, hi, best = 0.0, 10.0, (None, 10 ** 9)
    for _ in range(26):
        mid = (lo + hi) / 2
        n = count_fires(windows, detector, mid)
        if abs(n - n_target) < best[1]:
            best = (mid, abs(n - n_target))
        if n > n_target:
            lo = mid
        else:
            hi = mid
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windows_file")
    ap.add_argument("--detector", default="ks", choices=list(DETECTORS))
    ap.add_argument("--memory", type=int, default=3)
    args = ap.parse_args()

    w = load_windows(args.windows_file)
    base = majority_baseline(w)
    a, _ = run_policy(w, "never", memory=args.memory)

    print(f"{len(w)} windows, {len(w[0][0])} rows each")
    print(f"majority-class accuracy {base:.4f}; scores are AUC")
    print(f"never retrained: {a:.4f}\n")
    print(f"{'retrains':>9} {'B schedule':>12} {'C detector':>12} "
          f"{'C - B':>9}")
    print("-" * 48)

    rows = []
    for k in range(2, 13):
        b, nb = run_policy(w, "scheduled", k=k, memory=args.memory)
        thr = threshold_for_rate(w, args.detector, nb)
        c, nc = run_policy(w, "triggered", detector=args.detector,
                           threshold=thr, memory=args.memory)
        if nc != nb:
            note = f"  (detector fired {nc}, not {nb})"
        else:
            note = ""
        print(f"{nb:>9} {b:>12.4f} {c:>12.4f} {c-b:>+9.4f}{note}")
        if nc == nb:
            rows.append((nb, b, c))

    if not rows:
        print("\nno operating point where the counts matched exactly — "
              "read the table with the noted mismatches in mind")
        return

    best_b = max(rows, key=lambda r: r[1])
    best_c = max(rows, key=lambda r: r[2])
    print(f"\nbest schedule : {best_b[1]:.4f} at {best_b[0]} retrains")
    print(f"best detector : {best_c[2]:.4f} at {best_c[0]} retrains")
    print(f"difference    : {best_c[2] - best_b[1]:+.4f}")

    diffs = [c - b for _, b, c in rows]
    print(f"\nacross {len(rows)} exactly-matched operating points, "
          f"C - B ranges\n{min(diffs):+.4f} to {max(diffs):+.4f}, "
          f"mean {np.mean(diffs):+.4f}")
    if max(abs(np.mean(diffs)), 0) < 0.01:
        print("\nThe curves are effectively on top of each other: the signal")
        print("is worth nothing at ANY operating point, not merely at one.")
    elif np.mean(diffs) > 0:
        print("\nThe detector leads across operating points. Check the")
        print("persistence baseline before believing it — on an")
        print("autocorrelated target this can be momentum, not concept.")


if __name__ == "__main__":
    main()
