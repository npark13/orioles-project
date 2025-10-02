import pandas as pd
import numpy as np

# Haversine function
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth.
    Input: latitudes and longitudes in decimal degrees.
    Output: distance in kilometers.
    """
    R = 6371.0  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# Team locations (latitude, longitude) – replace with your full team mapping
team_coords = {
    "ARI": (33.4484, -112.0740),  # Phoenix
    "ATL": (33.7490, -84.3880),   # Atlanta
    "BAL": (39.2904, -76.6122),   # Baltimore
    "BOS": (42.3601, -71.0589),   # Boston
    "CHC": (41.8781, -87.6298),   # Chicago
    "CHW": (41.8781, -87.6298),   # Chicago
    "CIN": (39.1031, -84.5120),   # Cincinnati
    "CLE": (41.4993, -81.6944),   # Cleveland
    "COL": (39.7392, -104.9903),  # Denver
    "DET": (42.3314, -83.0458),   # Detroit
    "HOU": (29.7604, -95.3698),   # Houston
    "KCR": (39.0997, -94.5786),   # Kansas City
    "LAA": (33.8366, -117.9143),  # Anaheim
    "LAD": (34.0522, -118.2437),  # Los Angeles
    "MIA": (25.7617, -80.1918),   # Miami
    "MIL": (43.0389, -87.9065),   # Milwaukee
    "MIN": (44.9778, -93.2650),   # Minneapolis
    "NYM": (40.7128, -74.0060),   # New York
    "NYY": (40.7128, -74.0060),   # New York
    "OAK": (37.8044, -122.2711),  # Oakland
    "PHI": (39.9526, -75.1652),   # Philadelphia
    "PIT": (40.4406, -79.9959),   # Pittsburgh
    "SDP": (32.7157, -117.1611),  # San Diego
    "SEA": (47.6062, -122.3321),  # Seattle
    "SFG": (37.7749, -122.4194),  # San Francisco
    "STL": (38.6270, -90.1994),   # St. Louis
    "TBR": (27.9506, -82.4572),   # Tampa
    "TEX": (32.7555, -97.3308),   # Arlington
    "TOR": (43.6532, -79.3832),   # Toronto
    "WSN": (38.9072, -77.0369),   # Washington, D.C.
}

# Read plays.csv
plays = pd.read_csv("/Users/kevinhe/orioles-project/data/out/2024/plays.csv")

# Infer home and away teams from game_id
def infer_teams(game_id):
    # Assumes last 3 letters are home team, first 3 letters after initial are away
    home = game_id[-3:]
    away = game_id[3:6]
    return home, away

plays[['home_team', 'away_team']] = plays['game_id'].apply(lambda x: pd.Series(infer_teams(x)))

# Calculate distances per team
team_travel = {team: 0.0 for team in team_coords}

last_location = {team: None for team in team_coords}

for idx, row in plays.iterrows():
    for team in [row['home_team'], row['away_team']]:
        team_lat, team_lon = team_coords[team]
        if last_location[team] is not None:
            last_lat, last_lon = last_location[team]
            team_travel[team] += haversine(last_lat, last_lon, team_lat, team_lon)
        last_location[team] = (team_lat, team_lon)

# Output total travel distances per team
for team, dist in team_travel.items():
    print(f"{team}: {dist:.2f} km")