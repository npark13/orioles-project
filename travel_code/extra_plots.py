#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENERS_FILE = PROJECT_ROOT / "openers_significant_travel.csv"

if not OPENERS_FILE.exists():
    raise SystemExit("Missing openers_significant_travel.csv — run winning_vs_travel_2013_2024.py first.")

openers = pd.read_csv(OPENERS_FILE)

# --- 1) Travel Distance Histogram ---
fig, ax = plt.subplots(figsize=(6,4))
ax.hist(openers["visitor_travel_km"].dropna(), bins=30, edgecolor='black')
ax.set_title("Distribution of Visitor Travel Distance (Significant Trips Only)")
ax.set_xlabel("Distance Traveled (km)"); ax.set_ylabel("Number of Games")
fig.tight_layout(); fig.savefig(PROJECT_ROOT / "travel_distance_histogram.png", dpi=160)

# --- 2) Home Win vs Distance (continuous) ---
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(openers["visitor_travel_km"], openers["home_win"], alpha=0.25)
ax.set_xlabel("Visitor Travel Distance (km)"); ax.set_ylabel("Home Win (1 = Yes)")
ax.set_title("Home Win vs Travel Distance (Continuous, 2013–2024)")
fig.tight_layout(); fig.savefig(PROJECT_ROOT / "winning_vs_distance_scatter.png", dpi=160)
print("[OK] Saved histogram and scatter plots")

# --- 3) Home Win% vs Time-Zone Change Magnitude ---
if "tz_change" in openers.columns:
    tz_summary = openers.groupby("tz_change")["home_win"].mean().rename("home_win_pct").reset_index()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(tz_summary["tz_change"], tz_summary["home_win_pct"], marker="o")
    ax.set_title("Home Winning % vs Time-Zone Change Magnitude")
    ax.set_xlabel("HomeTZ − VisitorTZ (hours)"); ax.set_ylabel("Home Win %")
    ax.grid(True, alpha=0.4); fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "winning_vs_tzchange.png", dpi=160)

# --- 4) Angels Eastbound Trend by Year ---
if {"tz_change","visteam","date"}.issubset(openers.columns):
    openers["date"] = pd.to_datetime(openers["date"], errors="coerce")
    angels_east = openers[(openers["visteam"].isin(["ANA","LAA"])) & (openers["tz_change"] > 0)]
    if not angels_east.empty:
        angels_year = angels_east.groupby(angels_east["date"].dt.year)["home_win"].mean().rename("home_win_pct").reset_index()
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(angels_year["date"], angels_year["home_win_pct"], marker="o", color="orange")
        ax.set_title("Angels Eastbound Games – Opponent Home Win % by Year")
        ax.set_xlabel("Season"); ax.set_ylabel("Home Win % vs Angels (Eastbound)")
        ax.grid(True, alpha=0.4); fig.tight_layout()
        fig.savefig(PROJECT_ROOT / "angels_eastbound_trend.png", dpi=160)

print("Extra plots complete.")
