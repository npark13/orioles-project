# first_inning_bins.py
# Input: first_inning_features_with_tempo_all.csv
# Output: binned_first_inning_stats.csv, binned_first_inning_stats.png

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

IN = "first_inning_features_with_tempo_all.csv"   # built earlier
OUT_CSV = "binned_first_inning_stats.csv"
OUT_PNG = "binned_first_inning_stats.png"

# --- Load
df = pd.read_csv(IN)

# Pick the elapsed column (prefer filled, else raw)
elapsed_col = "top1_elapsed_seconds_final" if "top1_elapsed_seconds_final" in df.columns else "top1_elapsed_seconds"
if elapsed_col not in df.columns:
    raise SystemExit("No elapsed time column found (expected top1_elapsed_seconds_final or top1_elapsed_seconds).")

# Clean + derive target
d = df[[elapsed_col, "home_runs_bottom1", "season"]].copy()
d = d[pd.notna(d[elapsed_col])].copy()
# binary: did home score in B1?
d["home_scored_b1"] = (d["home_runs_bottom1"].fillna(0) > 0).astype(int)

# --- Bin elapsed seconds (quintiles)
# guard against ties by adding a tiny noise for qcut
_eps = np.random.default_rng(0).normal(0, 1e-6, len(d))
d["_elapsed_jitter"] = d[elapsed_col] + _eps
d["elapsed_bin"] = pd.qcut(d["_elapsed_jitter"], 5, labels=[1,2,3,4,5])  # 1=shortest, 5=longest
d.drop(columns=["_elapsed_jitter"], inplace=True)

# --- Aggregate overall
overall = (
    d.groupby("elapsed_bin", observed=True)
     .agg(
        games=("home_scored_b1","size"),
        pct_scored=("home_scored_b1","mean"),
        avg_elapse_sec=(elapsed_col,"mean")
     )
     .reset_index()
)
overall["pct_scored"] = overall["pct_scored"].astype(float)
overall.to_csv(OUT_CSV, index=False)

# --- (Optional) By season, for robustness
by_year = (
    d.groupby(["season","elapsed_bin"], observed=True)
     .agg(games=("home_scored_b1","size"),
          pct_scored=("home_scored_b1","mean"))
     .reset_index()
)

print("Rows used:", len(d))
print("\nOverall (quintiles shortest→longest):")
print(overall)

# --- Plot
fig, ax = plt.subplots(figsize=(7,4.5), dpi=160)
ax.plot(overall["elapsed_bin"].astype(int), overall["pct_scored"]*100, marker="D")
ax.set_xlabel("Top-1st Elapsed Time Quintile (1=shortest, 5=longest)")
ax.set_ylabel("Home Scored in Bottom-1st (%)")
ax.set_title("Bottom-1st Scoring vs Top-1st Elapsed Time (2015–present)")
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.set_xticks([1,2,3,4,5])

# annotate with mean elapsed time per bin
for x, y, sec in zip(overall["elapsed_bin"].astype(int), overall["pct_scored"]*100, overall["avg_elapse_sec"]):
    ax.text(x, y, f"\n{sec:.0f}s", ha="center", va="bottom", fontsize=9)

fig.tight_layout()
fig.savefig(OUT_PNG)
print(f"\nWrote: {OUT_CSV}")
print(f"Saved plot: {OUT_PNG}")
