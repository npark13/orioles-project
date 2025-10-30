#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd

# pip install pybaseball
from pybaseball import statcast

pd.options.mode.copy_on_write = True

OUT = Path(__file__).resolve().parents[1] / "statcast_pitches_raw_2018_2024.parquet"

def main():
    if len(sys.argv) != 3:
        print("Usage: python umpire_code/statcast_scrape.py YYYY-MM-DD YYYY-MM-DD")
        sys.exit(1)

    start, end = sys.argv[1], sys.argv[2]
    print(f"[scrape] Statcast {start}..{end} — this can take a while (minutes).")

    df = statcast(start_dt=start, end_dt=end)  # all MLB pitches
    if df is None or df.empty:
        print("[ERR] Statcast returned no rows. Try a smaller date range.")
        sys.exit(2)

    keep = [
        "game_date","game_pk","inning","inning_topbot","home_team","away_team",
        "pitch_type","pitch_name","description","type","balls","strikes","zone",
        "plate_x","plate_z","sz_bot","sz_top","stands","p_throws",
        "release_speed","release_spin_rate","release_extension"
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["plate_x","plate_z","sz_bot","sz_top"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"[OK] Saved {len(df):,} pitches → {OUT}")

if __name__ == "__main__":
    main()
