#!/usr/bin/env python3
# -------------------------------------------------------------
# winning_vs_travel_2013_2024.py
# Updated: filters to only first games after significant travel
# -------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER  = PROJECT_ROOT / "data" / "out"
RESULTS_CSV  = PROJECT_ROOT / "results_by_game.csv"
START_YEAR, END_YEAR = 2013, 2024

OUT_TRAVEL_CSV = PROJECT_ROOT / "winning_vs_travel_quartiles_2013_2024.csv"
OUT_TRAVEL_PNG = PROJECT_ROOT / "winning_vs_travel_2013_2024.png"
OUT_TZ_CSV     = PROJECT_ROOT / "winning_vs_tzdir_2013_2024.csv"
OUT_TZ_PNG     = PROJECT_ROOT / "winning_vs_tzdir_2013_2024.png"
OUT_ANGELS_CSV = PROJECT_ROOT / "angels_dir_compare.csv"
OUT_OPENERS    = PROJECT_ROOT / "openers_significant_travel.csv"

# ---------------- HELPERS ----------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

team_coords = {
    "ARI": (33.4484, -112.0740), "ATL": (33.7490, -84.3880), "BAL": (39.2904, -76.6122),
    "BOS": (42.3601, -71.0589), "CHN": (41.8781, -87.6298), "CHA": (41.8781, -87.6298),
    "CIN": (39.1031, -84.5120), "CLE": (41.4993, -81.6944), "COL": (39.7392, -104.9903),
    "DET": (42.3314, -83.0458), "HOU": (29.7604, -95.3698), "KCA": (39.0997, -94.5786),
    "LAA": (33.8366, -117.9143), "LAN": (34.0522, -118.2437), "MIA": (25.7617, -80.1918),
    "MIL": (43.0389, -87.9065), "MIN": (44.9778, -93.2650), "NYN": (40.7128, -74.0060),
    "NYA": (40.7128, -74.0060), "OAK": (37.8044, -122.2711), "PHI": (39.9526, -75.1652),
    "PIT": (40.4406, -79.9959), "SDN": (32.7157, -117.1611), "SEA": (47.6062, -122.3321),
    "SFN": (37.7749, -122.4194), "SLN": (38.6270, -90.1994), "TBA": (27.9506, -82.4572),
    "TEX": (32.7555, -97.3308), "TOR": (43.6532, -79.3832), "WAS": (38.9072, -77.0369)
}

TEAM_TZ = {
    "SEA": -8, "OAK": -8, "SFN": -8, "LAA": -8, "LAN": -8, "SDN": -8,
    "ARI": -7, "COL": -7,
    "KCA": -6, "MIN": -6, "TEX": -6, "HOU": -6, "CHN": -6, "CHA": -6, "MIL": -6,
    "DET": -5, "CLE": -5, "PIT": -5, "CIN": -5, "ATL": -5, "MIA": -5,
    "NYA": -5, "NYN": -5, "BOS": -5, "BAL": -5, "WAS": -5, "PHI": -5, "TBA": -5, "TOR": -5
}

def load_games_for_year(year: int) -> pd.DataFrame:
    fp = DATA_FOLDER / str(year) / "games.csv"
    if not fp.exists():
        return pd.DataFrame()
    g = pd.read_csv(fp, low_memory=False).rename(columns=str.lower)
    g["number"] = g.get("number", 1)
    g["site"] = g.get("site", g.get("park", g.get("hometeam")))
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    keep = [c for c in ["game_id","date","hometeam","visteam","site","number"] if c in g.columns]
    return g[keep].dropna(subset=["date"])

def compute_visitor_travel(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy().sort_values(["visteam","date","number"])
    last_loc, dists = {}, []
    for _, r in g.iterrows():
        v, h = r["visteam"], r["hometeam"]
        if v in team_coords and h in team_coords:
            dest = team_coords[h]
            d = haversine(*last_loc[v], *dest) if v in last_loc else np.nan
            last_loc[v] = dest
        else:
            d = np.nan
        dists.append(d)
    g["visitor_travel_km"] = dists
    return g

def flag_series_openers(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy().sort_values(["hometeam","visteam","site","date","number"])
    prev_key, prev_date, is_open = None, None, []
    for _, r in g.iterrows():
        key = (r["hometeam"], r["visteam"], r["site"])
        if key != prev_key:
            is_open.append(True)
        else:
            gap = (r["date"] - prev_date).days if pd.notna(prev_date) else 999
            is_open.append(gap > 1)
        prev_key, prev_date = key, r["date"]
    g["is_series_opener"] = is_open
    return g

def compute_tz_change(row):
    v, h = row["visteam"], row["hometeam"]
    if v not in TEAM_TZ or h not in TEAM_TZ: return 0
    return TEAM_TZ[h] - TEAM_TZ[v]

# ---------------- MAIN ----------------
def main():
    frames = [load_games_for_year(y) for y in range(START_YEAR, END_YEAR+1)]
    frames = [f for f in frames if not f.empty]
    if not frames: raise SystemExit("No games found.")
    games = pd.concat(frames, ignore_index=True)

    if not RESULTS_CSV.exists():
        raise SystemExit("Missing results_by_game.csv.")
    results = pd.read_csv(RESULTS_CSV)
    games = games.merge(results[["game_id","home_final","visitor_final"]], on="game_id", how="left")
    games = games.query("(home_final >= 0) & (visitor_final >= 0) & ((home_final + visitor_final) > 0)")
    games["home_win"] = (games["home_final"] > games["visitor_final"]).astype(int)

    games = compute_visitor_travel(games)
    games = flag_series_openers(games)
    games["tz_change"] = games.apply(compute_tz_change, axis=1)

    # Filter to first games after significant travel
    SIGNIFICANT_KM, SIGNIFICANT_TZ_DIFF = 1000, 1
    openers = games.query("is_series_opener == True").copy()
    openers = openers[
        (openers["visitor_travel_km"] >= SIGNIFICANT_KM)
        | (openers["tz_change"].abs() >= SIGNIFICANT_TZ_DIFF)
    ].dropna(subset=["visitor_travel_km"])
    openers["sig_travel"] = 1
    openers.to_csv(OUT_OPENERS, index=False)
    print(f"[OK] Wrote {OUT_OPENERS}")

    # Travel quartiles
    openers["travel_quartile"] = pd.qcut(openers["visitor_travel_km"], 4, labels=[1,2,3,4])
    summary = openers.groupby("travel_quartile")["home_win"].mean().rename("home_win_pct").reset_index()
    summary.to_csv(OUT_TRAVEL_CSV, index=False)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(summary["travel_quartile"].astype(int), summary["home_win_pct"], marker="D", s=70)
    ax.set_xticks([1,2,3,4])
    ax.set_xlabel("Travel Distance Quartile (Significant Travel Only)")
    ax.set_ylabel("Home Winning Percentage")
    ax.set_title("Winning vs Travel Distance (2013–2024)")
    ax.grid(True, axis="y", alpha=0.4)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT_TRAVEL_PNG)

    # Directional travel
    openers["dir_label"] = openers["tz_change"].apply(lambda t: "eastbound" if t>0 else "westbound" if t<0 else "same_zone")
    tz_summary = openers.groupby("dir_label")["home_win"].mean().rename("home_win_pct").reset_index()
    tz_summary.to_csv(OUT_TZ_CSV, index=False)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(tz_summary["dir_label"], tz_summary["home_win_pct"], color=["#a6cee3","#1f78b4","#b2df8a"])
    ax.set_xlabel("Travel Direction (TZ)")
    ax.set_ylabel("Home Winning Percentage")
    ax.set_title("Winning vs Time-Zone Direction (2013–2024)")
    ax.grid(True, axis="y", alpha=0.4)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT_TZ_PNG)

    # Angels case study
    openers["visitor_is_angels"] = openers["visteam"].isin(["ANA","LAA"]).astype(int)
    angels = (
        openers.groupby(["visitor_is_angels","dir_label"])["home_win"]
        .mean().rename("home_win_pct").reset_index()
    )
    angels.to_csv(OUT_ANGELS_CSV, index=False)

    print(f"[OK] Saved plots and CSVs to {PROJECT_ROOT}")

if __name__ == "__main__":
    main()
