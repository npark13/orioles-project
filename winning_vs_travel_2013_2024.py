# winning_vs_travel_2013_2024.py
# Produces: winning_vs_travel_quartiles_2013_2024.csv and winning_vs_travel_2013_2024.png

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
# Adjust paths if your repo layout differs
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FOLDER  = PROJECT_ROOT / "data" / "out"       # expects data/out/<YEAR>/games.csv
RESULTS_CSV  = PROJECT_ROOT / "results_by_game.csv"  # built from box scores
START_YEAR, END_YEAR = 2013, 2024

OUT_CSV = PROJECT_ROOT / "winning_vs_travel_quartiles_2013_2024.csv"
OUT_PNG = PROJECT_ROOT / "winning_vs_travel_2013_2024.png"

# ---------------- HELPERS ----------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# Approx stadium/city lat/lon by Retrosheet team code
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

def load_games_for_year(year: int) -> pd.DataFrame:
    """Load and normalize a year's games.csv."""
    fp = DATA_FOLDER / str(year) / "games.csv"
    if not fp.exists():
        return pd.DataFrame()
    g = pd.read_csv(fp, low_memory=False)
    g = g.rename(columns=str.lower)
    # normalize required columns
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
    """Distance the visitor traveled into each game from their previous game location."""
    games = games.copy().sort_values(["visteam","date","number"])
    last_loc = {}
    dists = []
    for _, r in games.iterrows():
        v = r["visteam"]; home = r["hometeam"]
        if home not in team_coords or v not in team_coords:
            dists.append(np.nan); continue
        dest = team_coords[home]
        if v in last_loc:
            d = haversine(*last_loc[v], *dest)
        else:
            d = np.nan
        dists.append(d)
        last_loc[v] = dest
    games["visitor_travel_km"] = dists
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

# ---------------- MAIN ----------------
def main():
    # Load all seasons
    frames = []
    for y in range(START_YEAR, END_YEAR + 1):
        gy = load_games_for_year(y)
        if not gy.empty:
            frames.append(gy)
    if not frames:
        raise SystemExit(f"No games found between {START_YEAR}-{END_YEAR} under {DATA_FOLDER}")

    games = pd.concat(frames, ignore_index=True)

    # Merge in final scores -> filter valid decisions -> home_win
    if not RESULTS_CSV.exists():
        raise SystemExit(f"Missing {RESULTS_CSV}. Build it from box scores first.")
    results = pd.read_csv(RESULTS_CSV)

    need_cols = {"game_id","home_final","visitor_final"}
    if need_cols - set(results.columns):
        raise SystemExit("results_by_game.csv must have columns: game_id, home_final, visitor_final")

    games = games.merge(results[list(need_cols)], on="game_id", how="left")

    # keep only games with an actual decision (no 0–0 placeholders)
    games = games[(games["home_final"].fillna(-1) >= 0) &
                (games["visitor_final"].fillna(-1) >= 0) &
                ((games["home_final"] + games["visitor_final"]) > 0)].copy()

    games["home_win"] = (games["home_final"] > games["visitor_final"]).astype(int)
    
    # Compute travel & flag series openers, then filter
    games = compute_visitor_travel(games)
    games = flag_series_openers(games)
    openers = games.query("is_series_opener == True").dropna(subset=["visitor_travel_km"]).copy()


    # Quartiles over the era
    openers["travel_quartile"] = pd.qcut(openers["visitor_travel_km"], 4, labels=[1,2,3,4])

    # Aggregate: home win% by quartile
    summary = (
        openers.groupby("travel_quartile", observed=True)["home_win"]
        .mean()
        .rename("home_win_pct")
        .reset_index()
    )
    summary["home_win_pct"] = summary["home_win_pct"].astype(float)

    # Save CSV
    summary.to_csv(OUT_CSV, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=160)

    # diamond markers ('D'); you can tweak s for size if you like (e.g., s=70)
    ax.scatter(
        summary["travel_quartile"].astype(int),
        summary["home_win_pct"],
        marker="D",  # ◇ diamond
        s=70
    )

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Travel Distance Quartile")
    ax.set_ylabel("Home Winning Percentage")
    ax.set_title("Winning and Travel, 2013–2024")

    # data-driven y-limits with small padding
    mn, mx = summary["home_win_pct"].min(), summary["home_win_pct"].max()
    pad = max(0.005, (mx - mn) * 0.3)
    ax.set_ylim(mn - pad, mx + pad)

    # make it look like your example: horizontal gridlines only
    ax.grid(True, axis="y", linewidth=1, alpha=0.4)
    ax.grid(False, axis="x")

    # optional: cleaner frame (like Excel charts)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f"[OK] Wrote {OUT_CSV}")
    print(f"[OK] Saved plot {OUT_PNG}")



if __name__ == "__main__":
    main()
