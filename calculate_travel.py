import pandas as pd
import numpy as np
from pathlib import Path

# Haversine function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# Stadium city coordinates (lat, lon)
team_coords = {
    "ARI": (33.4484, -112.0740),
    "ATL": (33.7490, -84.3880),
    "BAL": (39.2904, -76.6122),
    "BOS": (42.3601, -71.0589),
    "CHN": (41.8781, -87.6298),
    "CHA": (41.8781, -87.6298),
    "CIN": (39.1031, -84.5120),
    "CLE": (41.4993, -81.6944),
    "COL": (39.7392, -104.9903),
    "DET": (42.3314, -83.0458),
    "FLO": (25.7617, -80.1918),
    "HOU": (29.7604, -95.3698),
    "KCA": (39.0997, -94.5786),
    "ANA": (33.8366, -117.9143),
    "LAN": (34.0522, -118.2437),
    "MIA": (25.7617, -80.1918),
    "MIL": (43.0389, -87.9065),
    "MIN": (44.9778, -93.2650),
    "MON": (45.508889, -73.554167),
    "NYN": (40.7128, -74.0060),
    "NYA": (40.7128, -74.0060),
    "OAK": (37.8044, -122.2711),
    "PHI": (39.9526, -75.1652),
    "PIT": (40.4406, -79.9959),
    "SDN": (32.7157, -117.1611),
    "SEA": (47.6062, -122.3321),
    "SFN": (37.7749, -122.4194),
    "SLN": (38.6270, -90.1994),
    "TBA": (27.9506, -82.4572),
    "TEX": (32.7555, -97.3308),
    "TOR": (43.6532, -79.3832),
    "WAS": (38.9072, -77.0369),
}

# Path to game data folder
data_folder = Path("/Users/kevinhe/orioles-project/data/out")

# Track average travel per year
yearly_travel = []

for year in range(2000, 2025):
    games_file = data_folder / str(year) / "games.csv"
    if not games_file.exists():
        continue

    games = pd.read_csv(games_file)
    team_travel = {team: 0.0 for team in team_coords}
    last_location = {team: None for team in team_coords}

    for _, row in games.iterrows():
        home_team = row['hometeam']
        away_team = row['visteam']
        home_lat, home_lon = team_coords[home_team]

        # Home team travel
        if last_location[home_team] is not None:
            last_lat, last_lon = last_location[home_team]
            team_travel[home_team] += haversine(last_lat, last_lon, home_lat, home_lon)
        last_location[home_team] = (home_lat, home_lon)

        # Away team travel
        if last_location[away_team] is not None:
            last_lat, last_lon = last_location[away_team]
            team_travel[away_team] += haversine(last_lat, last_lon, home_lat, home_lon)
        last_location[away_team] = (home_lat, home_lon)

    # Compute average travel distance for this year
    avg_travel = np.mean(list(team_travel.values()))
    yearly_travel.append({"year": year, "avg_travel_km": round(avg_travel, 2)})

# Convert to DataFrame and save (no print)
yearly_travel_df = pd.DataFrame(yearly_travel)
yearly_travel_df.to_csv("average_travel_2020s.csv", index=False)