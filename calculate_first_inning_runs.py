from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import re

def runs_from_event(event: str) -> int:
    if not isinstance(event, str):
        return 0
    runs = len(re.findall(r"-H\b", event))
    if event.startswith("HR"):
        runs += 1
    return runs

def main():
    ap = argparse.ArgumentParser(description="Compute per-year first-inning runs summaries from plays.csv + games.csv")
    ap.add_argument("--out_root", default="data/out", help="Root folder that contains <year>/plays.csv and <year>/games.csv")
    ap.add_argument("--start_year", type=int, default=1911)
    ap.add_argument("--end_year", type=int, default=2024)
    args = ap.parse_args()

    out_root = Path(args.out_root)

    # Make relative paths behave like your other scripts: relative to CWD
    # (so run this from repo root)
    out_root = out_root if out_root.is_absolute() else (Path.cwd() / out_root)

    if not out_root.exists():
        raise SystemExit(f"[ERR] out_root does not exist: {out_root}")

    for year in range(args.start_year, args.end_year + 1):
        year_dir = out_root / str(year)
        plays_file = year_dir / "plays.csv"
        games_file = year_dir / "games.csv"
        output_file = year_dir / f"first_inning_runs_summary_{year}.csv"

        if not plays_file.exists() or not games_file.exists():
            # quiet skip is fine; keep your current behavior
            print(f"Skipping {year}, missing plays.csv or games.csv")
            continue

        plays_df = pd.read_csv(plays_file, low_memory=False)
        games_df = pd.read_csv(games_file, low_memory=False)

        first_inning = plays_df[plays_df["inning"] == 1.0].copy()
        first_inning["runs"] = first_inning["event_raw"].apply(runs_from_event)

        agg = (
            first_inning
            .groupby(["game_id", "batting_home"])["runs"]
            .sum()
            .reset_index()
        )

        pivot = agg.pivot(index="game_id", columns="batting_home", values="runs").reset_index()
        pivot = pivot.rename(columns={
            0: "visiting_first_inning_runs",
            1: "home_first_inning_runs"
        })

        merged = pivot.merge(
            games_df[["game_id", "visteam", "hometeam"]],
            on="game_id",
            how="left"
        )

        merged = merged[
            ["game_id", "hometeam", "visteam",
             "home_first_inning_runs", "visiting_first_inning_runs"]
        ]

        merged.to_csv(output_file, index=False)
        print(f"[OK] {year} → {output_file}")

if __name__ == "__main__":
    main()
