import pandas as pd
from pathlib import Path

root = Path("data/out/rolling_avg")

# ONLY per-year files: 2014, 2015, ..., 2024
files = sorted(root.glob("game_level_with_rolling_avg_weather_20??.csv"))
print("Found files:")
for fp in files:
    print("  -", fp.name)

base_cols = [
    "game_id",
    "date",
    "hometeam",
    "visteam",
    "home_first_inning_runs",
    "visiting_first_inning_runs",
    "home_ERA",
    "vis_ERA",
    "home_travel",
    "vis_travel",
    "home_last_location",
    "visiting_last_location",
    "home_avg_prev",
    "away_avg_prev",
    "home_OBP",
    "away_OBP",
]

dfs = []
for fp in files:
    print(f"\nReading {fp.name} ...")
    # Use the Python engine and be tolerant of weird extra columns
    df = pd.read_csv(
        fp,
        engine="python",       # more flexible than C engine
        on_bad_lines="warn",   # if a line has extra fields, skip it with a warning
    )

    # Keep only the canonical columns we care about
    cols_present = [c for c in base_cols if c in df.columns]
    missing = [c for c in base_cols if c not in df.columns]

    if missing:
        print(f"  [WARN] {fp.name} is missing columns: {missing}")
    else:
        print("  Columns OK")

    df = df[cols_present].copy()
    dfs.append(df)

# Concatenate all seasons
out = pd.concat(dfs, ignore_index=True)

out_path = root / "game_level_with_rolling_avg_weather_2014_2024_all_clean.csv"
out.to_csv(out_path, index=False)
print("\n[OK] Wrote", out_path)
print("Rows:", len(out))
