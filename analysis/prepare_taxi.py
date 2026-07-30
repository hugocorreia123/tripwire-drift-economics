#!/usr/bin/env python3
r"""Fetch and prepare NYC TLC yellow-taxi months for Tripwire.

Task: predict whether a trip earns a high tip. Real drift is available
here for free — COVID collapsed and reshaped tipping in 2020, and
congestion pricing arrived in January 2025 — and nobody designed either
to make a point, which is exactly why it beats a synthetic stream.

Two traps this handles, both of which would quietly wreck the study:

  1. TIP AMOUNT IS ONLY RECORDED FOR CARD PAYMENTS. Cash tips are not
     captured at all, so a cash trip looks like a zero-tip trip. Left
     in, the target is systematically wrong for a large share of rows,
     and the cash/card MIX itself drifts over time — which the model
     would learn as if it were tipping behaviour. Only card trips are
     kept.

  2. SCHEMA DRIFTS ACROSS YEARS. `cbd_congestion_fee` exists from 2025
     onward and `airport_fee` appears mid-series. Concatenating raw
     would introduce columns that are NaN for early months, and a drift
     detector would fire on the schema change rather than on the data.
     A fixed feature set present in every month is used instead.

    python3 analysis/prepare_taxi.py --months 2019-01:2021-12
    python3 analysis/prepare_taxi.py --months 2024-01:2025-12
"""
import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"

# present in every month across the range this targets
# PULocationID / DOLocationID are deliberately EXCLUDED: they are
# categorical zone identifiers, and passing them to a linear model as
# integers asserts that zone 132 is "more" than zone 43. They were in an
# earlier version and actively hurt. Hour and weekday replace them —
# tipping varies strongly by both, and they are genuinely ordinal-ish.
FEATURES = ["trip_distance", "passenger_count", "fare_amount",
            "extra", "mta_tax", "tolls_amount", "trip_seconds",
            "speed_mph", "hour", "weekday", "is_airport"]
NEEDED = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "tip_amount",
          "payment_type", "trip_distance", "passenger_count",
          "fare_amount", "extra", "mta_tax", "tolls_amount",
          "PULocationID", "DOLocationID"]


def months(spec):
    a, b = spec.split(":")
    ya, ma = (int(x) for x in a.split("-"))
    yb, mb = (int(x) for x in b.split("-"))
    out = []
    y, m = ya, ma
    while (y, m) <= (yb, mb):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def fetch(tag, cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"yellow_tripdata_{tag}.parquet"
    if f.exists():
        return f
    url = f"{BASE}/yellow_tripdata_{tag}.parquet"
    print(f"  downloading {tag} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, f)
        print(f"{f.stat().st_size/1e6:.0f} MB")
    except Exception as e:
        print(f"FAILED ({e})")
        if f.exists():
            f.unlink()
        return None
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", required=True,
                    help="range like 2024-01:2025-12")
    ap.add_argument("--per-month", type=int, default=20000,
                    help="rows sampled per month, keeping months balanced")
    ap.add_argument("--tip-threshold", type=float, default=0.20,
                    help="'high tip' = tip/fare above this ratio")
    ap.add_argument("--cache", default="data/raw")
    ap.add_argument("--out", default="data/taxi_prepared.parquet")
    args = ap.parse_args()

    frames = []
    for tag in months(args.months):
        f = fetch(tag, Path(args.cache))
        if f is None:
            continue
        df = pd.read_parquet(f, columns=[c for c in NEEDED])
        n0 = len(df)

        # trap 1: cash tips are never recorded
        df = df[df["payment_type"] == 1]
        # basic sanity: real trips, non-negative money
        df = df[(df["fare_amount"] > 2.5) & (df["trip_distance"] > 0)
                & (df["tip_amount"] >= 0) & (df["trip_distance"] < 100)]
        pick = pd.to_datetime(df["tpep_pickup_datetime"])
        df["trip_seconds"] = (
            pd.to_datetime(df["tpep_dropoff_datetime"]) - pick).dt.total_seconds()
        df = df[(df["trip_seconds"] > 30) & (df["trip_seconds"] < 7200)]
        df["speed_mph"] = df["trip_distance"] / (df["trip_seconds"] / 3600)
        df = df[(df["speed_mph"] > 0.5) & (df["speed_mph"] < 70)]
        df["hour"] = pick.dt.hour
        df["weekday"] = pick.dt.dayofweek
        # JFK (132) and LaGuardia (138) behave very differently on tips;
        # this is the one location fact worth keeping, as a flag rather
        # than as a magnitude
        df["is_airport"] = df["PULocationID"].isin([132, 138]).astype(int)

        if len(df) > args.per_month:
            df = df.sample(args.per_month, random_state=0)
        df["tip_ratio"] = df["tip_amount"] / df["fare_amount"]
        frames.append(df[["tpep_pickup_datetime", "tip_ratio"] + FEATURES])
        print(f"    {tag}: {n0:,} rows -> {len(df):,} kept "
              f"(card only, cleaned)")

    if not frames:
        sys.exit("no months downloaded")
    out = pd.concat(frames).sort_values("tpep_pickup_datetime")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)

    rate = (out["tip_ratio"] > args.tip_threshold).mean()
    if rate > 0.70 or rate < 0.30:
        print(f"\n  NOTE: base rate {rate:.3f} is heavily imbalanced. "
              f"Consider a\n  threshold nearer the median tip ratio "
              f"({out['tip_ratio'].median():.3f}) so the task has room\n"
              f"  to show degradation.")
    print(f"\nwrote {args.out}: {len(out):,} rows, "
          f"{len(frames)} months")
    print(f"high-tip base rate: {rate:.3f} "
          f"(tip/fare > {args.tip_threshold})")
    print(f"\nnext:\n  python3 analysis/real_stream.py {args.out} \\")
    print(f"    --time-col tpep_pickup_datetime --target tip_ratio \\")
    print(f"    --target-threshold {args.tip_threshold} --windows 40 \\")
    print(f"    --out data/windows.npz")
    print(f"  python3 analysis/run_comparison.py --real data/windows.npz")


if __name__ == "__main__":
    main()
