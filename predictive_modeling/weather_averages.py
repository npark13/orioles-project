import pandas as pd
import os

data_dir = "/Users/kevinhe/orioles-project/data/out"
output_dir = os.path.join(data_dir, "rolling_avg")
os.makedirs(output_dir, exist_ok=True)

years = range(2014, 2025)
home_run_col = "home_first_inning_runs"
away_run_col = "visiting_first_inning_runs"
n_games = 10

for year in years:
    file_path = os.path.join(data_dir, f"{year}/game_level_with_travel_{year}.csv")
    
    if not os.path.exists(file_path):
        print(f"File for {year} not found, skipping.")
        continue
    
    print(f"Processing year {year}...")
    df = pd.read_csv(file_path)
    
    # Sort by date so rolling averages are correct
    df = df.sort_values("date")
    
    df["home_avg_prev"] = pd.NA
    df["away_avg_prev"] = pd.NA
    
    teams = pd.unique(df[["hometeam", "visteam"]].values.ravel())
    
    for team in teams:
        # Home games
        home_games = df[df["hometeam"] == team]
        home_avg = (
            home_games[home_run_col]
            .shift(1)
            .rolling(window=n_games, min_periods=1)
            .mean()
        )
        # Fill first game NaN with the current game runs
        home_avg = home_avg.fillna(home_games[home_run_col])
        df.loc[home_games.index, "home_avg_prev"] = home_avg
        
        # Away games
        away_games = df[df["visteam"] == team]
        away_avg = (
            away_games[away_run_col]
            .shift(1)
            .rolling(window=n_games, min_periods=1)
            .mean()
        )
        away_avg = away_avg.fillna(away_games[away_run_col])
        df.loc[away_games.index, "away_avg_prev"] = away_avg
    
    # Save updated file
    output_path = os.path.join(output_dir, f"game_level_with_rolling_avg_weather_{year}.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved updated file to {output_path}")
