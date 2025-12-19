from pathlib import Path
import pandas as pd

def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "data" / "out").exists():
            return p
    raise SystemExit("Could not find repo root (expected to find data/out somewhere above this script).")

REPO_ROOT = Path(__file__).resolve().parents[1]
base_obp_path  = REPO_ROOT / "data" / "out"
base_game_path = REPO_ROOT / "data" / "out" / "rolling_avg"

years = range(2014, 2025)

for year in years:
    obp_file  = base_obp_path / str(year) / "team_obp.csv"
    game_file = base_game_path / f"game_level_with_rolling_avg_{year}.csv"

    if not obp_file.exists() or not game_file.exists():
        print(f"Skipping year {year}, file missing")
        continue

    print(f"Processing {year}...")

    df_obp   = pd.read_csv(obp_file)
    df_games = pd.read_csv(game_file)
    df_games = df_games.drop(columns=[c for c in ["home_OBP","away_OBP"] if c in df_games.columns])


    df_obp = df_obp[["team", "home", "away"]]
    df_obp_home = df_obp.rename(columns={"team": "hometeam", "home": "home_OBP"})
    df_obp_away = df_obp.rename(columns={"team": "visteam",  "away": "away_OBP"})

    df_games = df_games.merge(df_obp_home[["hometeam", "home_OBP"]], on="hometeam", how="left")
    df_games = df_games.merge(df_obp_away[["visteam", "away_OBP"]], on="visteam", how="left")

    df_games.to_csv(game_file, index=False)
    print(f"Updated {year} game-level file with home/away OBP")
