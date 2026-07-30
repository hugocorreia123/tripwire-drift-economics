#!/usr/bin/env python3
r"""Does the result survive the choices I made arbitrarily?

The ELEC2 frontier produced a mean C - B of +0.0212 across matched
operating points. That is one number from one stream under one set of
analysis choices — window count, detector, memory depth — none of which
were forced by the data. With a single dataset there is no honest
confidence interval to report, because the replicates would not be
independent.

What CAN be reported is sensitivity. If the sign of C - B flips when the
window count changes from 45 to 30, or when KS is swapped for PSI, then
the finding is an artefact of a choice and should not be published. If it
holds across every combination, it is worth something even without an
error bar.

This is a robustness grid, not a significance test, and it is labelled
as such.

    python3 analysis/robustness.py data/elec2_raw.csv
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.policies import run_policy, count_fires, DETECTORS
from analysis.prepare_elec2 import PREDICTORS


def windows_from_csv(path, n_windows):
    import pandas as pd
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns and "period" in df.columns:
        df = df.sort_values(["date", "period"]).reset_index(drop=True)
    cols = [c for c in PREDICTORS if c in df.columns]
    target = "class" if "class" in df.columns else df.columns[-1]
    yr = df[target].astype(str).str.lower()
    y = (yr == yr.value_counts().index[0]).astype(int).to_numpy()
    X = df[cols].to_numpy(dtype=float)
    ok = np.isfinite(X).all(axis=1)
    X, y = X[ok], y[ok]

    edges = np.linspace(0, len(X), n_windows + 1).astype(int)
    first = X[edges[0]:edges[1]]
    mu, sd = first.mean(axis=0), first.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    out = []
    for i in range(n_windows):
        lo, hi = edges[i], edges[i + 1]
        if hi - lo < 100 or len(np.unique(y[lo:hi])) < 2:
            continue
        out.append((Xs[lo:hi], y[lo:hi]))
    return out


def threshold_for_count(w, detector, n_target):
    lo, hi, best = 0.0, 10.0, (None, 10 ** 9)
    for _ in range(24):
        mid = (lo + hi) / 2
        n = count_fires(w, detector, mid)
        if abs(n - n_target) < best[1]:
            best = (mid, abs(n - n_target))
        if n > n_target:
            lo = mid
        else:
            hi = mid
    return best[0]


def mean_diff(w, detector, memory, ks=(4, 6, 8, 10)):
    """Mean C - B over operating points where the counts matched exactly."""
    diffs = []
    for k in ks:
        b, nb = run_policy(w, "scheduled", k=k, memory=memory)
        thr = threshold_for_count(w, detector, nb)
        c, nc = run_policy(w, "triggered", detector=detector,
                           threshold=thr, memory=memory)
        if nc == nb:
            diffs.append(c - b)
    return (float(np.mean(diffs)), len(diffs)) if diffs else (float("nan"), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    args = ap.parse_args()

    print("ROBUSTNESS GRID — is C - B an artefact of the analysis choices?")
    print("(mean over operating points where retrain counts matched exactly)")
    print("this is a sensitivity check, NOT a confidence interval\n")
    print(f"{'windows':>8} {'detector':>9} {'memory':>7} "
          f"{'mean C-B':>10} {'points':>7}")
    print("-" * 46)

    results = []
    for nw, det, mem in itertools.product((30, 45, 60), ("ks", "psi"), (2, 3, 5)):
        w = windows_from_csv(args.csv, nw)
        d, n = mean_diff(w, det, mem)
        if n == 0:
            print(f"{nw:>8} {det:>9} {mem:>7} {'no match':>10} {0:>7}")
            continue
        print(f"{nw:>8} {det:>9} {mem:>7} {d:>+10.4f} {n:>7}")
        results.append(d)

    if not results:
        print("\nno configuration produced matched operating points")
        return

    pos = sum(1 for d in results if d > 0)
    print(f"\n{len(results)} configurations: {pos} favour the detector, "
          f"{len(results)-pos} favour the schedule")
    print(f"range {min(results):+.4f} to {max(results):+.4f}, "
          f"mean {np.mean(results):+.4f}")

    if pos == len(results) or pos == 0:
        who = "detector" if pos else "schedule"
        print(f"\nThe sign is stable across every choice — the {who} leads")
        print("regardless of windowing, detector and memory. That is worth")
        print("reporting even without an error bar.")
    else:
        print("\nThe SIGN FLIPS across configurations. The result is an")
        print("artefact of choices that were not forced by the data, and")
        print("should not be reported as a finding.")


if __name__ == "__main__":
    main()
