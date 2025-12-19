#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# ----------------------------
# Paths (relative to repo root)
# ----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT  = REPO_ROOT / "data" / "out"
STADIUMS  = REPO_ROOT / "data" / "out" / "stadiums.csv"

YEARS = range(2014, 2025)
TRAVEL_FLOOR = 1e-6

# ----------------------------
# Load stadium coordinates
# ----------------------------
if not STADIUMS.exists():
    raise SystemExit(f"Missing stadiums.csv at {STADIUMS}")

stadiums = pd.read_csv(STADIUMS, low_memory=False)

# make sure lat/lon are numeric
stadiums["lat"] = pd.to_numeric(stadiums["lat"], errors="coerce")
stadiums["lon"] = pd.to_numeric(stadiums["lon"], errors="coerce")
stadiums = stadiums.dropna(subset=["team_id", "lat", "lon"])

stad_dict = stadiums.set_index("team_id")[["lat", "lon"]].to_dict(orient="index")

# ----------------------------
# Haversine distance (km)
# ----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# ----------------------------
# Main loop
# ----------------------------
for year in YEARS:
    year_dir = OUT_ROOT / str(year)
    in_file  = year_dir / f"first_inning_runs_with_era_{year}.csv"
    out_file = year_dir / f"game_level_with_travel_{year}.csv"

    print(f"\nProcessing {year}...")

    if not in_file.exists():
        print(f"  Skipping {year}: missing {in_file.name}")
        continue

    df = pd.read_csv(in_file, low_memory=False)

    # --- ensure date exists ---
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["game_id"].astype(str).str[3:11], format="%Y%m%d", errors="coerce")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)

    # ----------------------------
    # Track last location
    # ----------------------------
    df["home_last_location"] = "N/A"
    df["visiting_last_location"] = "N/A"

    last_loc = {}

    for i, row in df.iterrows():
        home = row["hometeam"]
        away = row["visteam"]

        if home in last_loc:
            df.at[i, "home_last_location"] = last_loc[home]
        if away in last_loc:
            df.at[i, "visiting_last_location"] = last_loc[away]

        # after this game, home team's last location = home stadium
        # away team's last location = this game's home stadium
        last_loc[home] = home
        last_loc[away] = home

    # ----------------------------
    # DROP games with no previous location
    # (first observed game for a team in this season)
    # ----------------------------
    before = len(df)
    df = df[(df["home_last_location"] != "N/A") & (df["visiting_last_location"] != "N/A")].copy()
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} games with missing previous location")

    if df.empty:
        print("  No games left after dropping first-location rows")
        continue

    # ----------------------------
    # Compute travel distances
    # ----------------------------
    home_travel = []
    vis_travel = []

    for _, row in df.iterrows():
        h_last = row["home_last_location"]
        v_last = row["visiting_last_location"]
        home   = row["hometeam"]

        try:
            home_travel.append(
                haversine(
                    float(stad_dict[h_last]["lat"]),
                    float(stad_dict[h_last]["lon"]),
                    float(stad_dict[home]["lat"]),
                    float(stad_dict[home]["lon"]),
                )
            )
            vis_travel.append(
                haversine(
                    float(stad_dict[v_last]["lat"]),
                    float(stad_dict[v_last]["lon"]),
                    float(stad_dict[home]["lat"]),
                    float(stad_dict[home]["lon"]),
                )
            )
        except Exception:
            home_travel.append(np.nan)
            vis_travel.append(np.nan)

    df["home_travel"] = pd.to_numeric(home_travel, errors="coerce")
    df["vis_travel"]  = pd.to_numeric(vis_travel, errors="coerce")

    # if any travel still missing (bad stadium mapping etc), drop those games too
    before2 = len(df)
    df = df.dropna(subset=["home_travel", "vis_travel"]).copy()
    dropped2 = before2 - len(df)
    if dropped2:
        print(f"  Dropped {dropped2} games with NaN travel (missing stadium coords?)")

    if df.empty:
        print("  No games left after dropping NaN travel rows")
        continue

    # ----------------------------
    # Enforce travel floor (0 / blanks / negatives -> 1e-6)
    # ----------------------------
    for col in ["home_travel", "vis_travel"]:
        df[col] = (
            df[col]
            .replace(r"^\s*$", np.nan, regex=True)
            .fillna(TRAVEL_FLOOR)
            .astype(float)
            .clip(lower=TRAVEL_FLOOR)
        )

    # ----------------------------
    # Enforce column order (travel cols last)
    # ----------------------------
    travel_cols = [
        "home_travel",
        "vis_travel",
        "home_last_location",
        "visiting_last_location",
    ]
    other_cols = [c for c in df.columns if c not in travel_cols]
    df = df[other_cols + travel_cols]

    # ----------------------------
    # Save
    # ----------------------------
    df.to_csv(out_file, index=False)
    print(f"  [OK] wrote {out_file} (rows={len(df)})")
