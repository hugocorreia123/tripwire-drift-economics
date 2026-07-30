#!/usr/bin/env python3
r"""Load several concept-drift benchmarks, not just one.

ELEC2 alone could not settle the question: its robustness grid flipped
sign across analysis choices, and its label is autocorrelated by
construction. One stream is one observation. The fix is more streams
with real degradation, run through identical machinery.

Each dataset here is standard in the streaming literature and reachable
from OpenML without credentials. For every one the loader reports the
two diagnostics that decide whether it can test anything at all:

  PERSISTENCE   accuracy of "predict the previous label". High values
                mean the target is autocorrelated and an apparent drift
                effect may be momentum. ELEC2 scores 0.848.

  DEGRADATION   whether a model trained once actually decays. Both taxi
                streams failed this — nothing degraded, so nothing could
                help, and the policy comparison was vacuous. A stream
                that does not degrade cannot test the hypothesis and is
                excluded rather than reported as a tie.

    python3 analysis/prepare_streams.py --list
    python3 analysis/prepare_streams.py airlines --windows 40
    python3 analysis/prepare_streams.py --all
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# OpenML names, with the columns to drop because they encode time
# directly — a model given the clock learns the calendar, not the market
CATALOGUE = {
    "electricity": dict(openml="electricity", version=1,
                        drop=["date", "day", "period"],
                        note="NSW electricity, the canonical benchmark"),
    "airlines": dict(openml="airlines", version=1,
                     drop=[],
                     note="flight delay, drift from schedule changes"),
    "covertype": dict(openml="covertype", version=3,
                      drop=[],
                      note="forest cover; drift is induced by ordering"),
    "poker": dict(openml="poker-hand", version=1,
                  drop=[],
                  note="poker hands; a common streaming benchmark"),
}


def fetch(name):
    spec = CATALOGUE[name]
    cache = Path(f"data/{name}_raw.csv")
    if cache.exists():
        import pandas as pd
        print(f"  using cached {cache}")
        return pd.read_csv(cache)
    from sklearn.datasets import fetch_openml
    print(f"  fetching '{spec['openml']}' from OpenML ...")
    d = fetch_openml(spec["openml"], version=spec["version"],
                     as_frame=True, parser="auto")
    df = d.frame.copy()
    df.columns = [c.lower() for c in df.columns]
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def to_windows(df, drop, n_windows, max_rows=60000):
    target = df.columns[-1]
    y_raw = df[target].astype(str)
    top = y_raw.value_counts().index[0]
    y = (y_raw == top).astype(int).to_numpy()

    feats = [c for c in df.columns
             if c != target and c not in drop
             and np.issubdtype(df[c].dtype, np.number)]
    if len(feats) < 2:
        # one-hot the small categoricals rather than give up
        import pandas as pd
        cats = [c for c in df.columns
                if c != target and c not in drop and df[c].nunique() <= 12]
        if cats:
            df = pd.concat([df, pd.get_dummies(df[cats], drop_first=True)],
                           axis=1)
            feats = [c for c in df.columns
                     if c != target and c not in drop and c not in cats
                     and np.issubdtype(df[c].dtype, np.number)]
    if len(feats) < 2:
        return None, None, f"only {len(feats)} usable numeric features"

    X = df[feats].to_numpy(dtype=float)
    ok = np.isfinite(X).all(axis=1)
    X, y = X[ok], y[ok]
    if len(X) > max_rows:               # keep the ORDER, thin evenly
        idx = np.linspace(0, len(X) - 1, max_rows).astype(int)
        X, y = X[idx], y[idx]

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
    return out, y, None


def diagnose(name, windows, y_full):
    """The two gates that decide whether a stream can test anything."""
    from analysis.policies import run_policy
    pers = float((y_full[1:] == y_full[:-1]).mean())
    a, _ = run_policy(windows, "never")
    b, _ = run_policy(windows, "scheduled", k=5)
    gain = b - a
    print(f"  windows      : {len(windows)} x {len(windows[0][0])} rows")
    print(f"  persistence  : {pers:.3f}"
          + ("   <- AUTOCORRELATED, treat results as suspect" if pers > 0.80
             else ""))
    print(f"  never / sched: {a:.4f} / {b:.4f}   retraining gains {gain:+.4f}")
    usable = gain > 0.02 and pers <= 0.90
    if gain <= 0.02:
        print("  VERDICT      : model does not degrade — cannot test the")
        print("                 hypothesis, exclude rather than report a tie")
    elif pers > 0.90:
        print("  VERDICT      : too autocorrelated to trust")
    else:
        print("  VERDICT      : USABLE — real degradation, acceptable "
              "autocorrelation")
    return usable, pers, gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--windows", type=int, default=40)
    args = ap.parse_args()

    if args.list:
        for k, v in CATALOGUE.items():
            print(f"  {k:12} {v['note']}")
        return

    names = list(CATALOGUE) if args.all else [args.name]
    if names == [None]:
        sys.exit("give a dataset name, or --all, or --list")

    summary = []
    for name in names:
        if name not in CATALOGUE:
            print(f"unknown dataset {name}")
            continue
        print(f"\n{'='*60}\n{name}\n{'='*60}")
        try:
            df = fetch(name)
        except Exception as e:
            print(f"  fetch failed: {str(e)[:120]}")
            continue
        w, y_full, err = to_windows(df, CATALOGUE[name]["drop"], args.windows)
        if err:
            print(f"  unusable: {err}")
            continue
        usable, pers, gain = diagnose(name, w, y_full)
        out = Path(f"data/{name}.npz")
        np.savez_compressed(
            out,
            **{f"X{i}": a for i, (a, b) in enumerate(w)},
            **{f"y{i}": b for i, (a, b) in enumerate(w)}, n=len(w))
        print(f"  wrote {out}")
        summary.append((name, usable, pers, gain))

    if len(summary) > 1:
        print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
        print(f"{'dataset':14} {'usable':>7} {'persist':>8} {'retrain gain':>13}")
        for n, u, p, g in summary:
            print(f"{n:14} {'yes' if u else 'no':>7} {p:>8.3f} {g:>+13.4f}")
        ok = [n for n, u, _, _ in summary if u]
        print(f"\n{len(ok)} of {len(summary)} streams can test the hypothesis"
              + (f": {', '.join(ok)}" if ok else ""))
        if ok:
            print("\nrun the robustness grid on each before believing any of them")


if __name__ == "__main__":
    main()
