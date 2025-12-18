#!/usr/bin/env python3
# -------------------------------------------------------------
# winning_vs_tzchange_2013_2024.py
# Minimal: builds winning_vs_tzchange.png only
# -------------------------------------------------------------

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER  = PROJECT_ROOT / "data" / "out"
RESULTS_CSV  = PROJECT_ROOT / "results_by_game.csv"

START_YEAR, END_YEAR = 2013, 2024

OUT_TZCHANGE_CSV = PROJECT_ROOT / "winning_vs_tzchange_2013_2024.csv"
OUT_TZCHANGE_PNG = PROJECT_ROOT / "winning_vs_tzchange.png"

# Filters (match your intent: "openers after significant TZ shift")
ONLY_SERIES_OPENERS = True
SIGNIFICANT_TZ_DIFF = 1  # keep games where abs(tz_change) >= 1

TEAM_TZ = {
    "SEA": -8, "OAK": -8, "SFN": -8, "LAA": -8, "LAN": -8, "SDN": -8,
    "ARI": -7, "COL": -7,
    "KCA": -6, "MIN": -6, "TEX": -6, "HOU": -6, "CHN": -6, "CHA": -6, "MIL": -6,
    "DET": -5, "CLE": -5, "PIT": -5, "CIN": -5, "ATL": -5, "MIA": -5,
    "NYA": -5, "NYN": -5, "BOS": -5, "BAL": -5, "WAS": -5, "PHI": -5, "TBA": -5, "TOR": -5
}

# ---------------- HELPERS ----------------
def load_games_for_year(year: int) -> pd.DataFrame:
    fp = DATA_FOLDER / str(year) / "games.csv"
    if not fp.exists():
        return pd.DataFrame()
    g = pd.read_csv(fp, low_memory=False).rename(columns=str.lower)

    g["number"] = g.get("number", 1)
    g["site"] = g.get("site", g.get("park", g.get("hometeam")))
    g["date"] = pd.to_datetime(g["date"], errors="coerce")

    keep = [c for c in ["game_id", "date", "hometeam", "visteam", "site", "number"] if c in g.columns]
    return g[keep].dropna(subset=["date"])

def flag_series_openers(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy().sort_values(["hometeam", "visteam", "site", "date", "number"])
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

def compute_tz_change(row) -> int:
    v, h = row["visteam"], row["hometeam"]
    if v not in TEAM_TZ or h not in TEAM_TZ:
        return 0
    return TEAM_TZ[h] - TEAM_TZ[v]

# ---------------- MAIN ----------------
def main():
    frames = [load_games_for_year(y) for y in range(START_YEAR, END_YEAR + 1)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit("No games found under data/out/<year>/games.csv")

    games = pd.concat(frames, ignore_index=True)

    if not RESULTS_CSV.exists():
        raise SystemExit(f"Missing {RESULTS_CSV}")

    results = pd.read_csv(RESULTS_CSV)
    games = games.merge(results[["game_id", "home_final", "visitor_final"]], on="game_id", how="left")

    # keep games with valid totals and not ties at 0-0 (matches your earlier filter)
    games = games.query("(home_final >= 0) & (visitor_final >= 0) & ((home_final + visitor_final) > 0)").copy()
    games.loc[:, "home_win"] = (games["home_final"] > games["visitor_final"]).astype(int)

    if ONLY_SERIES_OPENERS:
        games = flag_series_openers(games)
        games = games.query("is_series_opener == True").copy()

    games.loc[:, "tz_change"] = games.apply(compute_tz_change, axis=1)

    # significant tz change filter
    games = games[games["tz_change"].abs() >= SIGNIFICANT_TZ_DIFF].copy()
    if games.empty:
        raise SystemExit("No games left after tz_change filter — lower SIGNIFICANT_TZ_DIFF?")

    tz_summary = (
        games.groupby("tz_change", observed=True)["home_win"]
        .mean().rename("home_win_pct")
        .reset_index()
        .sort_values("tz_change")
    )
    tz_summary.to_csv(OUT_TZCHANGE_CSV, index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tz_summary["tz_change"], tz_summary["home_win_pct"], marker="o")
    ax.set_xlabel("Time-Zone Change (home TZ − visitor TZ)")
    ax.set_ylabel("Home Winning Percentage")
    ax.set_title("Winning vs Time-Zone Change (2013–2024)")
    ax.grid(True, axis="y", alpha=0.4)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_TZCHANGE_PNG, dpi=160)
    print(f"[OK] Wrote {OUT_TZCHANGE_CSV}")
    print(f"[OK] Saved {OUT_TZCHANGE_PNG}")

if __name__ == "__main__":
    main()
