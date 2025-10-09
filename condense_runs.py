import pandas as pd

# Input files
games_file = "/Users/kevinhe/orioles-project/data/out/2024/first_inning_runs_summary_2024.csv"
pitching_file = "/Users/kevinhe/orioles-project/data/out/2024/pitching_stats_fixed.csv"
output_file = "/Users/kevinhe/orioles-project/data/out/2024/first_inning_runs_with_era_2024.csv"

# Read CSVs
games_df = pd.read_csv(games_file)
pitching_df = pd.read_csv(pitching_file)

# Keep only Player ID and ERA
# Keep only Player ID and ERA
pitching_df = pitching_df[['Player-additional', 'p_era']].rename(
    columns={'Player-additional': 'pitcher_id', 'p_era': 'ERA'}
)

# Function to assign pitchers and ERAs
def assign_pitchers_era(row):
    if row['winner'] == row['hometeam']:
        home_pitcher_id = row['wp']
        vis_pitcher_id = row['lp']
    else:
        home_pitcher_id = row['lp']
        vis_pitcher_id = row['wp']
    
    # Lookup ERA
    home_ERA = pitching_df.loc[pitching_df['pitcher_id'] == home_pitcher_id, 'ERA'].values
    vis_ERA = pitching_df.loc[pitching_df['pitcher_id'] == vis_pitcher_id, 'ERA'].values
    
    # Assign values, handle missing ERA
    row['home_pitcher'] = home_pitcher_id
    row['vis_pitcher'] = vis_pitcher_id
    row['home_ERA'] = home_ERA[0] if len(home_ERA) > 0 else None
    row['vis_ERA'] = vis_ERA[0] if len(vis_ERA) > 0 else None
    
    return row

# Apply function
games_df = games_df.apply(assign_pitchers_era, axis=1)

# Select desired columns
columns_to_keep = [
    "game_id",
    "hometeam",
    "visteam",
    "home_first_inning_runs",
    "visiting_first_inning_runs",
    "home_pitcher",
    "vis_pitcher",
    "home_ERA",
    "vis_ERA"
]

df_final = games_df[columns_to_keep]

# Save to CSV
df_final.to_csv(output_file, index=False)

print(f"Condensed file with ERA saved to {output_file}")