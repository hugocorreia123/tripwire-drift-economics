#!/usr/bin/env python3
r"""Load ELEC2 (Australian NSW electricity market) as a Tripwire stream.

Why this dataset: the taxi runs showed spectacular covariate shift with
essentially no degradation, so they cannot discriminate between "the
drift signal is worthless" and "nothing was happening". ELEC2 is the
canonical concept-drift benchmark precisely because the price/demand
relationship is believed to move — 45,312 half-hourly records from
May 1996 to December 1998, labelled by whether price rose or fell
against a 24-hour moving average.

TWO CAVEATS, BUILT IN RATHER THAN DISCOVERED LATER.

The benchmark is contested. There is a paper titled "How good is the
Electricity benchmark for evaluating concept drift adaptation", and a
2025 analysis reports an error trace with no major changes, contrary to
the classic Gama result. Treat a finding here as one more data point,
not as settlement.

The label is AUTOCORRELATED by construction. It compares price against
a trailing 24-hour average, so consecutive records are strongly
dependent and a "predict the previous label" rule scores far above
chance without learning anything. This module measures that persistence
baseline explicitly: if it rivals the model, any apparent drift effect
may be autocorrelation rather than concept change, and the result
should not be trusted.

Standard protocol (following the streaming literature) excludes the
date, day and period columns and predicts from the market variables
alone — otherwise the model learns the clock rather than the market.

    python3 analysis/prepare_elec2.py --windows 45 --out data/elec2.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np


PREDICTORS = ["nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]


def load_elec2():
    """Fetch from OpenML. Falls back to a local CSV if one is present,
    so the experiment stays reproducible offline once fetched."""
    local = Path("data/elec2_raw.csv")
    if local.exists():
        import pandas as pd
        print(f"using cached {local}")
        return pd.read_csv(local)

    from sklearn.datasets import fetch_openml
    print("fetching 'electricity' from OpenML ...")
    d = fetch_openml("electricity", version=1, as_frame=True, parser="auto")
    df = d.frame.copy()
    df.columns = [c.lower() for c in df.columns]
    local.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(local, index=False)
    print(f"cached to {local}")
    return df


def persistence_baseline(y):
    """Accuracy of simply predicting the previous label.

    On a target defined against a trailing average this can be very
    high, and it learns nothing. If it approaches what the model
    achieves, the stream is autocorrelation-dominated and a drift
    result from it is not credible.
    """
    return float((y[1:] == y[:-1]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=45)
    ap.add_argument("--out", default="data/elec2.npz")
    args = ap.parse_args()

    df = load_elec2()
    cols = [c for c in PREDICTORS if c in df.columns]
    if len(cols) < 4:
        sys.exit(f"expected market columns {PREDICTORS}, found {list(df.columns)}")

    # time order is essential and the file ships ordered; date+period
    # reconstruct it explicitly rather than trusting row order
    if "date" in df.columns and "period" in df.columns:
        df = df.sort_values(["date", "period"]).reset_index(drop=True)

    target = "class" if "class" in df.columns else df.columns[-1]
    y_raw = df[target].astype(str).str.lower()
    y = (y_raw == y_raw.value_counts().index[0]).astype(int).to_numpy()
    X = df[cols].to_numpy(dtype=float)

    ok = np.isfinite(X).all(axis=1)
    X, y = X[ok], y[ok]

    pers = persistence_baseline(y)
    print(f"\nrows            : {len(X):,}")
    print(f"predictors      : {cols}")
    print(f"positive rate   : {y.mean():.3f}")
    print(f"persistence     : {pers:.3f}  <- accuracy of 'predict the "
          f"previous label'")
    if pers > 0.80:
        print("\n  *** The label is strongly autocorrelated. A model that")
        print("  *** appears to learn may be tracking persistence, and any")
        print("  *** drift effect measured here should be treated as")
        print("  *** suspect. This is the documented critique of ELEC2.")

    # standardise on the first window only — global statistics would
    # leak future distribution information backwards
    edges = np.linspace(0, len(X), args.windows + 1).astype(int)
    first = X[edges[0]:edges[1]]
    mu, sd = first.mean(axis=0), first.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd

    windows = []
    for i in range(args.windows):
        lo, hi = edges[i], edges[i + 1]
        yi = y[lo:hi]
        if hi - lo < 100 or len(np.unique(yi)) < 2:
            continue
        windows.append((Xs[lo:hi], yi))

    rates = [w[1].mean() for w in windows]
    print(f"\nwindows         : {len(windows)}, {len(windows[0][0])} rows each")
    print(f"positive rate   : {min(rates):.3f} to {max(rates):.3f} "
          f"(first {rates[0]:.3f}, last {rates[-1]:.3f})")
    if max(rates) - min(rates) > 0.15:
        print("  -> the base rate moves substantially across the stream,")
        print("     which is real prior drift")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        **{f"X{i}": w[0] for i, w in enumerate(windows)},
        **{f"y{i}": w[1] for i, w in enumerate(windows)},
        n=len(windows))
    print(f"\nwrote {args.out}")
    print(f"now run: python3 analysis/run_comparison.py --real {args.out}")


if __name__ == "__main__":
    main()
