from pathlib import Path
import os
import pandas as pd

# -------------------------------------------------
# Force working directory to repo root
# -------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)

# -------------------------------------------------
# Paths (repo-root based)
# -------------------------------------------------
DATA_OUT = REPO_ROOT / "data" / "out"
ROLLING_OUT = DATA_OUT / "rolling_avg"
ROLLING_OUT.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Parameters
# -------------------------------------------------
years = range(2014, 2025)
home_run_col = "home_first_inning_runs"
away_run_col = "visiting_first_inning_runs"
n_games = 10

# -------------------------------------------------
# Main loop
# -------------------------------------------------
for year in years:
    input_path = DATA_OUT / str(year) / f"game_level_with_travel_{year}.csv"

    if not input_path.exists():
        print(f"Skipping {year}: {input_path.name} not found")
        continue

    print(f"Processing year {year}...")
    df = pd.read_csv(input_path)

    # Ensure correct ordering for rolling stats
    df = df.sort_values("date").reset_index(drop=True)

    df["home_avg_prev"] = pd.NA
    df["away_avg_prev"] = pd.NA

    teams = pd.unique(df[["hometeam", "visteam"]].values.ravel())

    for team in teams:
        # ----------------------------
        # Home games rolling average
        # ----------------------------
        home_games = df[df["hometeam"] == team]
        home_avg = (
            home_games[home_run_col]
            .shift(1)
            .rolling(window=n_games, min_periods=1)
            .mean()
        )
        home_avg = home_avg.fillna(home_games[home_run_col])
        df.loc[home_games.index, "home_avg_prev"] = home_avg

        # ----------------------------
        # Away games rolling average
        # ----------------------------
        away_games = df[df["visteam"] == team]
        away_avg = (
            away_games[away_run_col]
            .shift(1)
            .rolling(window=n_games, min_periods=1)
            .mean()
        )
        away_avg = away_avg.fillna(away_games[away_run_col])
        df.loc[away_games.index, "away_avg_prev"] = away_avg

    # -------------------------------------------------
    # Save output
    # -------------------------------------------------
    output_path = ROLLING_OUT / f"game_level_with_rolling_avg_{year}.csv"
    df.to_csv(output_path, index=False)
    print(f"  [OK] wrote {output_path}")
