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
    # add more as needed
}

# Known hyphenated first names
hyphenated_firsts = {
    "Hyun Jin": "Hyun-Jin",
    # add more as needed
}

# Normalize function
def normalize_name(s):
    if isinstance(s, str):
        s = s.strip()

        # Ignore "League Average" only
        if s.lower() == "league average":
            return None

        # Handle "Last, First" format
        if "," in s:
            last, first = s.split(",", 1)
            s = f"{first.strip()} {last.strip()}"

        # Remove accents
        s = ''.join(c for c in unicodedata.normalize('NFD', s)
                    if unicodedata.category(c) != 'Mn')

        # Remove apostrophes and non-letter chars except space and hyphen
        s = re.sub(r"[^A-Za-z -]", "", s)

        # Remove common suffixes: Jr., Sr., II, III, IV, V, etc.
        s = re.sub(r"\b(Jr|Sr|I{2,3}|IV|V)\b", "", s, flags=re.IGNORECASE).strip()

        # Split parts and fix nicknames
        parts = s.split()
        if parts and parts[0] in nickname_map:
            parts[0] = nickname_map[parts[0]]
        s = ' '.join(parts)

        # Replace known hyphenated first names
        for key, val in hyphenated_firsts.items():
            s = s.replace(key, val)

        return s.lower()  # lowercase for case-insensitive matching
    return s

# Loop through years
for year in range(2014, 2025):
    pitching_path = pitching_folder.format(year=year)
    rosters_path = rosters_folder.format(year=year)
    output_path = output_folder.format(year=year)

    if not os.path.exists(pitching_path) or not os.path.exists(rosters_path):
        print(f"Skipping year {year}: file(s) not found.")
        continue

    # Load CSVs
    pitching_df = pd.read_csv(pitching_path)
    rosters_df = pd.read_csv(rosters_path)

    # Normalize names
    pitching_df['player_name_clean'] = pitching_df['Player'].apply(normalize_name)
    rosters_df['player_name_clean'] = rosters_df['player_name'].apply(normalize_name)

    # Remove rows where normalization returned None (like "League Average")
    pitching_df = pitching_df[pitching_df['player_name_clean'].notna()].copy()

    # Map normalized names to player_id
    player_id_map = dict(zip(rosters_df['player_name_clean'], rosters_df['player_id']))

    # Replace Player-additional with player_id
    pitching_df['Player-additional'] = pitching_df['player_name_clean'].map(player_id_map)

    # Optional: print unmatched players
    missing = pitching_df[pitching_df['Player-additional'].isna()]
    if not missing.empty:
        print(f"Year {year} - Players not found in rosters:")
        print(missing[['Player']])

    # Drop helper column before saving
    pitching_df = pitching_df.drop(columns=['player_name_clean'])

    # Write fixed CSV
    pitching_df.to_csv(output_path, index=False)
    print(f"Year {year} - Fixed CSV written to {output_path}")