import pandas as pd
import os
import sys

# Base directory paths
base_dir = "/Users/kevinhe/orioles-project/data/out"

# --- Handle command-line year argument ---
if len(sys.argv) >= 2:
    year = int(sys.argv[1])
else:
    print("Usage: python condense_runs.py <year>")
    sys.exit(1)

print(f"Processing {year}...")

# Input files
games_file = f"{base_dir}/{year}/first_inning_runs_summary_{year}.csv"
pitching_file = f"{base_dir}/{year}/pitching_stats_fixed.csv"
output_file = f"{base_dir}/{year}/first_inning_runs_with_era_{year}.csv"

# Skip year if files are missing
if not (os.path.exists(games_file) and os.path.exists(pitching_file)):
    print(f"Missing files for {year}, skipping.")
    sys.exit(0)

# Read CSVs
games_df = pd.read_csv(games_file)
pitching_df = pd.read_csv(pitching_file)

# Ensure ERA column exists
if 'ERA' not in pitching_df.columns:
    print(f"'ERA' not found in pitching file for {year}, columns: {pitching_df.columns.tolist()}")
    sys.exit(1)

# Keep only Player ID and ERA
pitching_df = pitching_df[['Player-additional', 'ERA']].rename(
    columns={'Player-additional': 'pitcher_id', 'ERA': 'ERA'}
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
    row['year'] = year

    return row

# Apply function
games_df = games_df.apply(assign_pitchers_era, axis=1)

# Select desired columns
columns_to_keep = [
    "year",
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

# Save output
df_final.to_csv(output_file, index=False)
print(f"Saved {output_file}")
