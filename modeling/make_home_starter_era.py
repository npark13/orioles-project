#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick builder for data/out/home_starter_era.csv

Computes team-level home ERA proxy (RA/9) from Retrosheet-style games.csv files.
Each row = hometeam, season, home_era
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def make_home_starter_era(games_root: Path, start: int, end: int):
    rows = []
    for y in range(start, end + 1):
        fp = games_root / str(y) / "games.csv"
        if not fp.exists():
            print(f"⚠️ Missing {fp}")
            continue
        g = pd.read_csv(fp, low_memory=False)
        g.columns = [c.lower() for c in g.columns]

        if "date" not in g.columns or "hometeam" not in g.columns or "visteam" not in g.columns:
            print(f"⚠️ Skipping {y}, missing key columns.")
            continue

        g["date"] = pd.to_datetime(g["date"], errors="coerce")
        g["season"] = g["date"].dt.year

        # Identify visitor runs column
        vis_cols = [c for c in g.columns if "vis" in c and "run" in c]
        if not vis_cols:
            print(f"⚠️ Couldn’t find visitor runs column in {fp}")
            continue
        vcol = vis_cols[0]

        # Compute total visitor runs per team per season (proxy for home ERA)
        agg = g.groupby(["hometeam", "season"], as_index=False)[vcol].sum()
        games_ct = g.groupby(["hometeam", "season"], as_index=False).size().rename(columns={"size": "games"})
        merged = agg.merge(games_ct, on=["hometeam", "season"], how="inner")
        merged["home_era"] = merged[vcol] / (merged["games"] * 9.0) * 9.0  # runs per 9 innings
        rows.append(merged[["hometeam", "season", "home_era"]])

    if not rows:
        print("No data collected.")
        return pd.DataFrame(columns=["hometeam", "season", "home_era"])

    df = pd.concat(rows, ignore_index=True)
    return df.groupby(["hometeam", "season"], as_index=False)["home_era"].mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-root", type=str, default="data/out")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--out", type=str, default="data/out/home_starter_era.csv")
    args = ap.parse_args()

    games_root = Path(args.games_root)
    out_fp = Path(args.out)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    df = make_home_starter_era(games_root, args.start, args.end)
    df.to_csv(out_fp, index=False)
    print(f"✅ saved {out_fp} with {len(df)} rows")

if __name__ == "__main__":
    main()

