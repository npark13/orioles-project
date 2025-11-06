import pandas as pd
import os

# Base paths
base_obp_path = "/Users/kevinhe/orioles-project/data/out"
base_game_path = "/Users/kevinhe/orioles-project/data/out/rolling_avg"

years = range(2014, 2025)  # 2014 through 2024

for year in years:
    obp_file = os.path.join(base_obp_path, f"{year}", "team_obp.csv")
    game_file = os.path.join(base_game_path, f"game_level_with_rolling_avg_weather_{year}.csv")
    
    # Check if files exist
    if not os.path.exists(obp_file) or not os.path.exists(game_file):
        print(f"Skipping year {year}, file missing")
        continue
    
    # Read CSVs
    df_obp = pd.read_csv(obp_file)
    df_games = pd.read_csv(game_file)
    
    # Keep only relevant columns
    df_obp = df_obp[['team', 'home', 'away']]
    
    # Prepare home and away OBP mappings
    df_obp_home = df_obp.rename(columns={'team': 'hometeam', 'home': 'home_OBP'})
    df_obp_away = df_obp.rename(columns={'team': 'visteam', 'away': 'away_OBP'})
    
    # Merge home and away OBP
    df_games = df_games.merge(df_obp_home[['hometeam', 'home_OBP']], on='hometeam', how='left')
    df_games = df_games.merge(df_obp_away[['visteam', 'away_OBP']], on='visteam', how='left')
    
    # Save updated CSV (overwrite same file)
    df_games.to_csv(game_file, index=False)
    print(f"Updated {year} game-level file with home/away OBP")
