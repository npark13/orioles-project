import pandas as pd
import unicodedata
import re
import os

# Folder paths
pitching_folder = "/Users/kevinhe/orioles-project/data/out/{year}/pitching_stats.csv"
rosters_folder = "/Users/kevinhe/orioles-project/data/out/{year}/rosters.csv"
output_folder = "/Users/kevinhe/orioles-project/data/out/{year}/pitching_stats_fixed.csv"

# Nickname mapping
nickname_map = {
    "Mike": "Michael",
    "Mikey": "Michael",
    "Alex": "Alexander",
    "Andy": "Andrew",
    "Will": "William",
    "Bill": "William",
    "Matt": "Matthew",
    "Jake": "Jakob",
}

# Known hyphenated first names
hyphenated_firsts = {
    "Hyun Jin": "Hyun-Jin",
}

# Function to normalize names
def normalize_name(s):
    if isinstance(s, str):
        s = s.strip()
        if s.lower() == "league average":
            return None
        # Remove accents
        s = ''.join(c for c in unicodedata.normalize('NFD', s)
                    if unicodedata.category(c) != 'Mn')
        # Remove non-letter chars except space and hyphen
        s = re.sub(r"[^A-Za-z -]", "", s)
        # Remove suffixes
        s = re.sub(r"\b(Jr|Sr|I{2,3}|IV|V)\b", "", s, flags=re.IGNORECASE).strip()
        # Fix nicknames
        parts = s.split()
        if parts and parts[0] in nickname_map:
            parts[0] = nickname_map[parts[0]]
        s = ' '.join(parts)
        # Hyphenated first names
        for key, val in hyphenated_firsts.items():
            s = s.replace(key, val)
        return s.lower()
    return s

# Loop through all years
for year in range(2015, 2025):
    pitching_path = pitching_folder.format(year=year)
    rosters_path = rosters_folder.format(year=year)
    output_path = output_folder.format(year=year)

    if not os.path.exists(pitching_path) or not os.path.exists(rosters_path):
        print(f"Skipping year {year}: file(s) not found.")
        continue

    # Load CSVs
    pitching_df = pd.read_csv(pitching_path, quotechar='"')
    rosters_df = pd.read_csv(rosters_path, quotechar='"')

    # Strip spaces from column names
    pitching_df.columns = [c.strip() for c in pitching_df.columns]
    rosters_df.columns = [c.strip() for c in rosters_df.columns]

    # Combine last_name + first_name into "Player" if separate columns exist
    if 'last_name' in pitching_df.columns and 'first_name' in pitching_df.columns:
        pitching_df['Player'] = pitching_df['first_name'].astype(str) + ' ' + pitching_df['last_name'].astype(str)
    elif 'last_name, first_name' in pitching_df.columns:
        # Handle older CSV format
        pitching_df.rename(columns={'last_name, first_name': 'Player'}, inplace=True)
        pitching_df['Player'] = pitching_df['Player'].apply(
            lambda x: ' '.join([y.strip() for y in x.split(',')[::-1]]) if pd.notna(x) else x
        )
    else:
        print(f"Year {year}: Could not find name columns in pitching CSV")
        continue

    # Normalize names
    pitching_df['player_name_clean'] = pitching_df['Player'].apply(normalize_name)
    rosters_df['player_name_clean'] = rosters_df['player_name'].apply(normalize_name)

    # Drop rows where normalization returned None
    pitching_df = pitching_df[pitching_df['player_name_clean'].notna()].copy()

    # Map normalized names to player_id
    player_id_map = dict(zip(rosters_df['player_name_clean'], rosters_df['player_id']))
    pitching_df['Player-additional'] = pitching_df['player_name_clean'].map(player_id_map)

    # Print unmatched players
    missing = pitching_df[pitching_df['Player-additional'].isna()]
    if not missing.empty:
        print(f"Year {year} - Players not found in rosters:")
        print(missing[['Player']])

    # Drop helper column
    pitching_df = pitching_df.drop(columns=['player_name_clean'])

    # Save fixed CSV
    pitching_df.to_csv(output_path, index=False)
    print(f"Year {year} - Fixed CSV written to {output_path}")