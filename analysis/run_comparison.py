#!/usr/bin/env python3
"""Run the matched-frequency policy comparison across drift regimes.

    python3 analysis/run_comparison.py            # all regimes
    python3 analysis/run_comparison.py concept_only
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.drift_stream import DriftStream, REGIMES
from analysis.policies import run_policy, calibrate_threshold

K, N_SEEDS = 5, 8


def one(name):
    A, B, C, nb, nc = [], [], [], [], []
    for seed in range(N_SEEDS):
        s = DriftStream(seed=seed, **REGIMES[name])
        w = [(X, y) for _, X, y in s.stream()]
        a, _ = run_policy(w, "never")
        b, rb = run_policy(w, "scheduled", k=K)
        thr = calibrate_threshold(w, "ks", 1.0 / K)
        c, rc = run_policy(w, "triggered", detector="ks", threshold=thr)
        A.append(a); B.append(b); C.append(c); nb.append(rb); nc.append(rc)
    d = np.array(C) - np.array(B)
    se = d.std(ddof=1) / np.sqrt(len(d))
    star = "*" if abs(d.mean()) > 1.96 * se else " "
    print(f"{name:16} {np.mean(A):>7.3f} {np.mean(B):>8.3f} {np.mean(C):>8.3f} "
          f"{d.mean():>+8.4f} {star}  retrains {np.mean(nb):.1f}/{np.mean(nc):.1f}")


if __name__ == "__main__":
    which = sys.argv[1:] or list(REGIMES)
    print(f"matched retrain rate 1/{K}; prequential accuracy; "
          f"{N_SEEDS} seeds")
    print(f"{'regime':16} {'A never':>7} {'B sched':>8} {'C trig':>8} "
          f"{'C - B':>8}")
    print("-" * 68)
    for n in which:
        one(n)
    print("\nC - B is the value of the SIGNAL with retraining frequency held")
    print("equal. * marks a difference beyond 1.96 standard errors.")
