#!/usr/bin/env python3
r"""Turn a real timestamped table into the window stream Tripwire expects.

Stage 0 ran on a synthetic stream whose drift regimes I chose. That is
the right way to validate the machinery — the answer is known in advance
— but it is not evidence about the world, and a skeptic could fairly say
the detector was set up to fail. Real drift settles it.

This produces exactly the `[(X, y), ...]` format `run_comparison.py`
already consumes, so nothing downstream changes.

Requirements of the input file: one timestamp column, one binary target
(or a numeric column plus --target-threshold to binarise), and numeric
features. Rows are ordered by time and cut into equal-count windows, so
each window carries the same statistical weight regardless of how
unevenly the data is spread across the calendar.

    python3 analysis/real_stream.py data.parquet \
        --time-col tpep_pickup_datetime --target tip_amount \
        --target-threshold 2.0 --windows 40
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def build_windows(df, time_col, target, features=None, n_windows=40,
                  target_threshold=None, max_rows=None, min_window=200):
    """Equal-count windows ordered by time.

    Equal-COUNT rather than equal-duration: a calendar month with three
    times the traffic would otherwise dominate, and the drift detector
    would react to sample size instead of distribution.
    """
    df = df.dropna(subset=[time_col, target]).copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    if max_rows and len(df) > max_rows:
        # keep the time ordering; sample evenly rather than truncating,
        # which would silently drop the most recent drift
        idx = np.linspace(0, len(df) - 1, max_rows).astype(int)
        df = df.iloc[idx]

    y_raw = df[target].to_numpy()
    if target_threshold is not None:
        y = (y_raw > target_threshold).astype(int)
    else:
        uniq = np.unique(y_raw[~pd.isna(y_raw)])
        if len(uniq) != 2:
            raise SystemExit(
                f"target '{target}' has {len(uniq)} distinct values; pass "
                f"--target-threshold to binarise it")
        y = (y_raw == uniq.max()).astype(int)

    if features is None:
        drop = {time_col, target}
        features = [c for c in df.columns
                    if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    if not features:
        raise SystemExit("no numeric feature columns found")

    X = df[features].to_numpy(dtype=float)
    ok = np.isfinite(X).all(axis=1)
    X, y = X[ok], y[ok]
    # standardise on the FIRST window only — using global statistics
    # would leak future distribution information backwards and mask the
    # very drift this experiment measures
    edges = np.linspace(0, len(X), n_windows + 1).astype(int)
    first = X[edges[0]:edges[1]]
    mu, sd = first.mean(axis=0), first.std(axis=0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd

    windows = []
    for i in range(n_windows):
        lo, hi = edges[i], edges[i + 1]
        if hi - lo < min_window:
            continue
        yi = y[lo:hi]
        if len(np.unique(yi)) < 2:      # a model cannot be fit here
            continue
        windows.append((X[lo:hi], yi))
    return windows, features


def describe(windows, features):
    print(f"windows        : {len(windows)}")
    print(f"rows/window    : {len(windows[0][0])}")
    print(f"features       : {len(features)}  {features[:6]}"
          + (" ..." if len(features) > 6 else ""))
    rates = [w[1].mean() for w in windows]
    print(f"positive rate  : {min(rates):.3f} to {max(rates):.3f} "
          f"(first {rates[0]:.3f}, last {rates[-1]:.3f})")
    if abs(rates[-1] - rates[0]) > 0.05:
        print("  -> the base rate itself moves across the stream, which is")
        print("     real prior drift and exactly what makes this a fair test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--time-col", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--target-threshold", type=float, default=None)
    ap.add_argument("--features", default=None,
                    help="comma-separated; default is every numeric column")
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--max-rows", type=int, default=200_000)
    ap.add_argument("--out", default="windows.npz")
    args = ap.parse_args()

    p = Path(args.path)
    df = (pd.read_parquet(p) if p.suffix in (".parquet", ".pq")
          else pd.read_csv(p))
    print(f"loaded {len(df):,} rows from {p.name}")

    feats = args.features.split(",") if args.features else None
    windows, used = build_windows(
        df, args.time_col, args.target, feats, args.windows,
        args.target_threshold, args.max_rows)
    if len(windows) < 10:
        raise SystemExit(f"only {len(windows)} usable windows — need >=10")
    describe(windows, used)

    np.savez_compressed(
        args.out,
        **{f"X{i}": w[0] for i, w in enumerate(windows)},
        **{f"y{i}": w[1] for i, w in enumerate(windows)},
        n=len(windows))
    print(f"\nwrote {args.out}")
    print(f"now run: python3 analysis/run_comparison.py --real {args.out}")


def load_windows(path):
    """Read back what this script wrote."""
    z = np.load(path)
    return [(z[f"X{i}"], z[f"y{i}"]) for i in range(int(z["n"]))]


if __name__ == "__main__":
    main()
