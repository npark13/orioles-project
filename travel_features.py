from __future__ import annotations
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path


TEAM_CITIES = {
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

