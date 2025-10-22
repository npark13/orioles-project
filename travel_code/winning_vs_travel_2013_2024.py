#!/usr/bin/env python3
"""
winning_vs_travel_2013_2024.py

Builds two summaries for *series openers* across 2013–2024:
1) Distance quartiles (visitor travel into the opener) vs home win%
2) Time-zone direction (eastbound / westbound / same) vs home win%

Outputs in project root:
  - winning_vs_travel_quartiles_2013_2024.csv / .png
  - winning_vs_tzdir_2013_2024.csv / .png
  - angels_dir_compare.csv (Angels vs league, by direction)

Requires:
  data/out/<YEAR>/games.csv  (Retrosheet-derived; must include game_id, date, hometeam, visteam, etc.)
  results_by_game.csv        (at project root; columns: game_id, home_final, visitor_final)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from zoneinfo import ZoneInfo

# ---------- Project-aware paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER  = PROJECT_ROOT / "data" / "out"        # expects data/out/<YEAR>/games.csv
RESULTS_CSV  = PROJECT_ROOT / "results_by_game.csv" # built from box scores
START_YEAR, END_YEAR = 2013, 2024

OUT_CSV_DIST = PROJECT_ROOT / "winning_vs_travel_quartiles_2013_2024.csv"
OUT_PNG_DIST = PROJECT_ROOT / "winning_vs_travel_2013_2024.png"
OUT_CSV_DIR  = PROJECT_ROOT / "winning_vs_tzdir_2013_2024.csv"
OUT_PNG_DIR  = PROJECT_ROOT / "winning_vs_tzdir_2013_2024.png"
OUT_ANGELS   = PROJECT_ROOT / "angels_dir_compare.csv"

# ---------- Coords by Retrosheet-ish code (approx city centers) ----------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

team_coords = {
    "ARI": (33.4484, -112.0740), "ATL": (33.7490, -84.3880),  "BAL": (39.2904, -76.6122),
    "BOS": (42.3601, -71.0589),  "CHN": (41.8781, -87.6298),  "CHA": (41.8781, -87.6298),
    "CIN": (39.1031, -84.5120),  "CLE": (41.4993, -81.6944),  "COL": (39.7392, -104.9903),
    "DET": (42.3314, -83.0458),  "FLA": (25.7617, -80.1918),  "MIA": (25.7617, -80.1918),
    "HOU": (29.7604, -95.3698),  "KCA": (39.0997, -94.5786),  "ANA": (33.8366, -117.9143),
    "LAA": (33.8366, -117.9143), "LAN": (34.0522, -118.2437), "MIL": (43.0389, -87.9065),
    "MIN": (44.9778, -93.2650),  "MON": (45.5089, -73.5542),  "NYN": (40.7128, -74.0060),
    "NYA": (40.7128, -74.0060),  "OAK": (37.8044, -122.2711), "PHI": (39.9526, -75.1652),
    "PIT": (40.4406, -79.9959),  "SDN": (32.7157, -117.1611), "SEA": (47.6062, -122.3321),
    "SFN": (37.7749, -122.4194), "SLN": (38.6270, -90.1994),  "TBA": (27.9506, -82.4572),
    "TEX": (32.7555, -97.3308),  "TOR": (43.6532, -79.3832),  "WAS": (38.9072, -77.0369),
}

# ---------- Home park time zones (IANA names; handles DST via zoneinfo) ----------
TEAM_TZ = {
    "ARI": "America/Phoenix",
    "ATL": "America/New_York",
    "BAL": "America/New_York",
    "BOS": "America/New_York",
    "CHN": "America/Chicago",   # Cubs
    "CHA": "America/Chicago",   # White Sox
    "CIN": "America/New_York",
    "CLE": "America/New_York",
    "COL": "America/Denver",
    "DET": "America/Detroit",
    "FLA": "America/New_York",
    "MIA": "America/New_York",
    "HOU": "America/Chicago",
    "KCA": "America/Chicago",
    "ANA": "America/Los_Angeles",
    "LAA": "America/Los_Angeles",
    "LAN": "America/Los_Angeles",
    "MIL": "America/Chicago",
    "MIN": "America/Chicago",
    "MON": "America/Toronto",
    "NYN": "America/New_York",
    "NYA": "America/New_York",
    "OAK": "America/Los_Angeles",
    "PHI": "America/New_York",
    "PIT": "America/New_York",
    "SDN": "America/Los_Angeles",
    "SEA": "America/Los_Angeles",
    "SFN": "America/Los_Angeles",
    "SLN": "America/Chicago",
    "TBA": "America/New_York",
    "TEX": "America/Chicago",
    "TOR": "America/Toronto",
    "WAS": "America/New_York",
}

def utc_offset_hours(tz_name: str, date_ts: pd.Timestamp) -> int:
    """Return integer UTC offset in hours for a given local wall-clock date."""
    tz = ZoneInfo(tz_name)
    localized = date_ts.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    return int(localized.utcoffset().total_seconds() // 3600)

# ---------- IO ----------
def load_games_for_year(year: int) -> pd.DataFrame:
    fp = DATA_FOLDER / str(year) / "games.csv"
    if not fp.exists():
        return pd.DataFrame()
    g = pd.read_csv(fp, low_memory=False).rename(columns=str.lower)
    if "number" not in g.columns:
        g["number"] = 1
    if "site" not in g.columns and "park" in g.columns:
        g["site"] = g["park"]
    if "site" not in g.columns:
        g["site"] = g["hometeam"]
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    keep = [c for c in ["game_id","date","hometeam","visteam","site","number"] if c in g.columns]
    g = g[keep].dropna(subset=["date"])
    return g

def compute_visitor_travel(games: pd.DataFrame) -> pd.DataFrame:
    """Distance (km) and TZ direction (east/west/same) the visitor traveled into each game."""
    games = games.copy().sort_values(["visteam","date","number"])
    last_loc = {}
    last_tz = {}
    dists, tz_deltas, east, west, abs_tz = [], [], [], [], []

    for _, r in games.iterrows():
        v = r["visteam"]; home = r["hometeam"]

        # distance
        if home in team_coords and v in team_coords and v in last_loc:
            d = haversine(*last_loc[v], *team_coords[home])
        else:
            d = np.nan
        dists.append(d)
        if home in team_coords:
            last_loc[v] = team_coords[home]

        # tz direction
        tz_delta = np.nan
        if home in TEAM_TZ:
            curr_tz = TEAM_TZ[home]
            curr_off = utc_offset_hours(curr_tz, r["date"])
            if v in last_tz:
                prev_off = utc_offset_hours(last_tz[v], r["date"])
                tz_delta = curr_off - prev_off
            last_tz[v] = curr_tz

        tz_deltas.append(tz_delta)
        east.append(1 if pd.notna(tz_delta) and tz_delta > 0 else 0)
        west.append(1 if pd.notna(tz_delta) and tz_delta < 0 else 0)
        abs_tz.append(abs(tz_delta) if pd.notna(tz_delta) else np.nan)

    games["visitor_travel_km"] = dists
    games["tz_delta"] = tz_deltas
    games["eastbound"] = east
    games["westbound"] = west
    games["abs_tz"] = abs_tz
    return games

def flag_series_openers(g: pd.DataFrame) -> pd.DataFrame:
    """Mark first game of a home series vs a given visitor at a park."""
    g = g.copy().sort_values(["hometeam","visteam","site","date","number"])
    is_open = []
    prev_key = None; prev_date = None
    for _, r in g.iterrows():
        key = (r["hometeam"], r["visteam"], r["site"])
        if key != prev_key:
            is_open.append(True)
        else:
            gap = (r["date"] - prev_date).days if pd.notna(prev_date) else 999
            is_open.append(gap > 1)
        prev_key = key; prev_date = r["date"]
    g["is_series_opener"] = is_open
    return g

# ---------- MAIN ----------
def main():
    # load seasons
    frames = []
    for y in range(START_YEAR, END_YEAR + 1):
        gy = load_games_for_year(y)
        if not gy.empty:
            frames.append(gy)
    if not frames:
        raise SystemExit(f"No games found between {START_YEAR}-{END_YEAR} under {DATA_FOLDER}")

    games = pd.concat(frames, ignore_index=True)

    # results (scores) -> home_win
    if not RESULTS_CSV.exists():
        raise SystemExit(f"Missing {RESULTS_CSV}. Build it from box scores first.")
    results = pd.read_csv(RESULTS_CSV)

    need_cols = {"game_id","home_final","visitor_final"}
    if need_cols - set(results.columns):
        raise SystemExit("results_by_game.csv must have columns: game_id, home_final, visitor_final")

    games = games.merge(results[list(need_cols)], on="game_id", how="left")
    games = games[(games["home_final"].fillna(-1) >= 0) &
                  (games["visitor_final"].fillna(-1) >= 0) &
                  ((games["home_final"] + games["visitor_final"]) > 0)].copy()
    games["home_win"] = (games["home_final"] > games["visitor_final"]).astype(int)

    # travel features + series openers
    games = compute_visitor_travel(games)
    games = flag_series_openers(games)
    openers = games.query("is_series_opener == True").dropna(subset=["visitor_travel_km"]).copy()

    # ---------- (1) Distance quartiles summary ----------
    openers["travel_quartile"] = pd.qcut(openers["visitor_travel_km"], 4, labels=[1,2,3,4])
    summary_dist = (
        openers.groupby("travel_quartile", observed=True)["home_win"]
               .mean()
               .rename("home_win_pct")
               .reset_index()
    )
    summary_dist["home_win_pct"] = summary_dist["home_win_pct"].astype(float)
    summary_dist.to_csv(OUT_CSV_DIST, index=False)

    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=160)
    ax.scatter(summary_dist["travel_quartile"].astype(int), summary_dist["home_win_pct"], marker="D", s=70)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Visitor Travel Distance Quartile")
    ax.set_ylabel("Home Winning Percentage")
    ax.set_title("Winning vs Visitor Travel Distance (Series Openers, 2013–2024)")
    mn, mx = summary_dist["home_win_pct"].min(), summary_dist["home_win_pct"].max()
    pad = max(0.005, (mx - mn) * 0.3)
    ax.set_ylim(mn - pad, mx + pad)
    ax.grid(True, axis="y", linewidth=1, alpha=0.4); ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT_PNG_DIST)
    print(f"[OK] Wrote {OUT_CSV_DIST}")
    print(f"[OK] Saved plot {OUT_PNG_DIST}")

    # ---------- (2) Time-zone direction summary ----------
    openers["dir_label"] = np.select(
        [openers["eastbound"].eq(1), openers["westbound"].eq(1)],
        ["eastbound","westbound"],
        default="same_zone"
    )
    summary_dir = (
        openers.dropna(subset=["dir_label"])
               .groupby("dir_label", observed=True)["home_win"]
               .mean()
               .rename("home_win_pct")
               .reset_index()
    )
    summary_dir.to_csv(OUT_CSV_DIR, index=False)

    fig2, ax2 = plt.subplots(figsize=(6, 4.2), dpi=160)
    x = np.arange(len(summary_dir))
    ax2.scatter(x, summary_dir["home_win_pct"], marker="D", s=70)
    ax2.set_xticks(x); ax2.set_xticklabels(summary_dir["dir_label"])
    ax2.set_ylabel("Home Winning Percentage")
    ax2.set_title("Winning vs Visitor Time-Zone Direction (Series Openers, 2013–2024)")
    ax2.grid(True, axis="y", linewidth=1, alpha=0.4); ax2.grid(False, axis="x")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    fig2.tight_layout(); fig2.savefig(OUT_PNG_DIR)
    print(f"[OK] Wrote {OUT_CSV_DIR}")
    print(f"[OK] Saved plot {OUT_PNG_DIR}")

    # ---------- (3) Angels vs league (by direction) ----------
    ang = openers.assign(visitor_is_angels=openers["visteam"].isin(["LAA","ANA"]).astype(int))
    angels_vs_league = (
        ang.groupby(["visitor_is_angels", "dir_label"], observed=True)["home_win"]
           .mean().rename("home_win_pct").reset_index()
    )
    angels_vs_league.to_csv(OUT_ANGELS, index=False)
    print(f"[OK] Wrote {OUT_ANGELS}")

if __name__ == "__main__":
    main()
