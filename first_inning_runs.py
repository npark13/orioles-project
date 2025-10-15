import pandas as pd
import unicodedata
import re
import os
import sys

# --- Folder paths ---
pitching_folder = "/Users/kevinhe/orioles-project/data/out/{year}/pitching_stats.csv"
rosters_folder = "/Users/kevinhe/orioles-project/data/out/{year}/rosters.csv"
output_folder = "/Users/kevinhe/orioles-project/data/out/{year}/pitching_stats_fixed.csv"

# --- Nickname and variant mappings ---
nickname_map = {
    "Mike": "Michael",
    "Mikey": "Michael",
    "Alex": "Alexander",
    "Andy": "Andrew",
    "Will": "William",
    "Bill": "William",
    "Matt": "Matthew",
    "Jake": "Jakob",
    "Louie": "Louis",
    "Jose": "Jose",
    "Kike": "Enrique",
    "Enrique": "Enrique",
    "Guillermo":"Guillo",
    "Jake": "Jacob",
}

# Hyphenated first names
hyphenated_firsts = {
    "Hyun Jin": "Hyun-Jin",
    "Jong Ho": "Jong-Ho",
    "Choi Ji": "Choi-Ji",
}

# Special full-name replacements
special_case_names = {
    "louis varland": "louie varland",
    "jose a ferrer": "jose ferrer",
    "enrique hernandez": "kike hernandez",
}

# --- Normalize player name ---
def normalize_name(s):
    if not isinstance(s, str):
        return s

    if "," in s:
        last, first = s.split(",", 1)
        s = f"{first.strip()} {last.strip()}"

    s = ''.join(c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn')

    s = re.sub(r"\b([A-Z])[\.\s]*([A-Z])[\.\s]*\b", r"\1\2", s)
    s = re.sub(r"[^A-Za-z -]", "", s)
    s = re.sub(r"\b(Jr|Sr|I{2,3}|IV|V)\b", "", s, flags=re.IGNORECASE).strip()

    for key, val in hyphenated_firsts.items():
        s = s.replace(key, val)

    parts = s.split()
    if parts and parts[0] in nickname_map:
        parts[0] = nickname_map[parts[0]]
    s = ' '.join(parts).lower().strip()

    if s in special_case_names:
        s = special_case_names[s]

    return s

# --- Handle year argument ---
if len(sys.argv) >= 2:
    year = int(sys.argv[1])
else:
    year = 2024  # default year

print(f"\n=== Processing year {year} ===\n")

pitching_path = pitching_folder.format(year=year)
rosters_path = rosters_folder.format(year=year)
output_path = output_folder.format(year=year)

if not os.path.exists(pitching_path) or not os.path.exists(rosters_path):
    print(f"Skipping year {year}: file(s) not found.")
    sys.exit(0)

# --- Load CSVs ---
pitching_df = pd.read_csv(pitching_path)
rosters_df = pd.read_csv(rosters_path)

# Normalize names
pitching_df['player_name_clean'] = pitching_df['Player'].apply(normalize_name)
rosters_df['player_name_clean'] = rosters_df['player_name'].apply(normalize_name)

# Map player names to IDs
player_id_map = dict(zip(rosters_df['player_name_clean'], rosters_df['player_id']))
pitching_df['Player-additional'] = pitching_df['player_name_clean'].map(player_id_map)

# Report missing players
missing = pitching_df[pitching_df['Player-additional'].isna()]
if not missing.empty:
    print(f"Year {year} — Players not found in roster:")
    print(missing[['Player']])
    print()

# Drop helper column
pitching_df = pitching_df.drop(columns=['player_name_clean'])

# Save fixed CSV
pitching_df.to_csv(output_path, index=False)
print(f"Year {year} — Fixed CSV written to {output_path}")