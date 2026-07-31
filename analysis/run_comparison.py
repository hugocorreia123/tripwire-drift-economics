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


def real(path):
    """Same three policies, on a real stream instead of a synthetic one.
    Only one 'seed' exists — the data — so the seed-level error bar is
    replaced by a note, not silently omitted."""
    from analysis.real_stream import load_windows
    from analysis.policies import majority_baseline, persistence_auc
    w = load_windows(path)
    base = majority_baseline(w)
    pers = persistence_auc(w)
    a, _ = run_policy(w, "never")
    b, rb = run_policy(w, "scheduled", k=K)
    thr = calibrate_threshold(w, "ks", 1.0 / K)
    c, rc = run_policy(w, "triggered", detector="ks", threshold=thr)
    # On a single stream the calibration lands only approximately, and a
    # retrain-count mismatch hands one policy free compute — the exact
    # confound this design exists to remove. So also run the schedule at
    # C's OBSERVED rate and compare against that.
    k_match = max(1, round((len(w) - 1) / rc)) if rc else K
    b2, rb2 = run_policy(w, "scheduled", k=k_match)

    print(f"real data: {len(w)} windows, {len(w[0][0])} rows each")
    print(f"majority-class accuracy would be {base:.4f}; scores below "
          f"are AUC (0.5 = worthless)")
    print(f"{'P persistence':>16} {pers:.4f}   <- predicts the last label, "
          f"learns nothing")
    print(f"{'A never':>16} {a:.4f}")
    print(f"{'B scheduled':>16} {b:.4f}   ({rb} retrains, k={K})")
    print(f"{'C triggered':>16} {c:.4f}   ({rc} retrains, "
          f"threshold {thr:.4f})")
    print(f"{'B rate-matched':>16} {b2:.4f}   ({rb2} retrains, k={k_match})")
    print(f"\n  C - B (nominal k)      = {c-b:+.4f}"
          + ("" if rb == rc else f"   <- {rc} vs {rb} retrains, not comparable"))
    print(f"  C - B (rate-matched)   = {c-b2:+.4f}"
          + ("   <- use this one" if rb2 == rc else
             f"   <- {rc} vs {rb2} retrains, still off by {abs(rc-rb2)}"))
    if not (pers != pers) and pers > max(a, b, c) - 0.02:
        print("\n  *** The persistence reference matches or beats every")
        print("  *** retraining policy. This stream's structure is MOMENTUM,")
        print("  *** not concept drift, and any apparent advantage for the")
        print("  *** detector is an artefact of label stickiness.")
    if max(a, b, c) < 0.58:
        print("\n  *** AUC is near 0.5: the model has almost no predictive")
        print("  *** power on this task, so there is nothing for drift to")
        print("  *** degrade and nothing for retraining to recover. This run")
        print("  *** does not test the hypothesis — fix the task, not the")
        print("  *** policies.")
    print("  One dataset means one observation: this cannot carry an error")
    print("  bar the way the synthetic runs do. Read it as corroboration or")
    print("  contradiction of the synthetic result, not as a measurement.")


if __name__ == "__main__":
    if "--real" in sys.argv:
        real(sys.argv[sys.argv.index("--real") + 1])
        sys.exit(0)
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
