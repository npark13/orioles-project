import pandas as pd
import re
import os

# Function to count runs from an event
def runs_from_event(event: str) -> int:
    if not isinstance(event, str):
        return 0
    runs = len(re.findall(r'-H\b', event))
    if event.startswith("HR"):
        runs += 1
    return runs

# Base data folder
base_folder = "/Users/kevinhe/orioles-project/data/out/"

# Loop explicitly over years 1911-2024
for year in range(1911, 2025):
    year_folder = os.path.join(base_folder, str(year))
    plays_file = os.path.join(year_folder, "plays.csv")
    games_file = os.path.join(year_folder, "games.csv")
    output_file = os.path.join(year_folder, f"first_inning_runs_summary_{year}.csv")

    # Check if both files exist
    if not os.path.exists(plays_file) or not os.path.exists(games_file):
        print(f"Skipping {year}, missing plays.csv or games.csv")
        continue

    # Load data
    plays_df = pd.read_csv(plays_file, low_memory=False)
    games_df = pd.read_csv(games_file, low_memory=False)

    # Filter for first inning
    first_inning = plays_df[plays_df['inning'] == 1.0].copy()

    # Count runs
    first_inning.loc[:, 'runs'] = first_inning['event_raw'].apply(runs_from_event)

    # Aggregate runs by game_id and batting_home (0 = away, 1 = home)
    agg = first_inning.groupby(['game_id', 'batting_home']).agg({'runs': 'sum'}).reset_index()

    # Pivot to get home vs away columns
    pivot = agg.pivot(index='game_id', columns='batting_home', values='runs').reset_index()
    pivot = pivot.rename(columns={0: 'visiting_first_inning_runs', 1: 'home_first_inning_runs'})

    # Merge with games.csv to get team IDs
    merged = pivot.merge(games_df[['game_id', 'visteam', 'hometeam']], on='game_id', how='left')

    # Reorder columns for output
    merged = merged[['game_id', 'hometeam', 'visteam', 'home_first_inning_runs', 'visiting_first_inning_runs']]

    # Save to CSV
    merged.to_csv(output_file, index=False)
    print(f"Saved first-inning runs summary for {year} to {output_file}")