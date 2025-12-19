# first_inning_bins.py
# Input: first_inning_features_with_tempo_all.csv
# Output: binned_first_inning_stats.csv

import pandas as pd
import numpy as np
from pathlib import Path

IN = Path("csv_files") / "first_inning_features_with_tempo_all.csv"
OUT_CSV = Path("csv_files") / "binned_first_inning_stats.csv"

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# --- Load
df = pd.read_csv(IN)

# Pick the elapsed column (prefer filled, else raw)
elapsed_col = (
    "top1_elapsed_seconds_final"
    if "top1_elapsed_seconds_final" in df.columns
    else "top1_elapsed_seconds"
)
if elapsed_col not in df.columns:
    raise SystemExit(
        "No elapsed time column found (expected top1_elapsed_seconds_final or top1_elapsed_seconds)."
    )

# Clean + derive target
d = df[[elapsed_col, "home_runs_bottom1", "season"]].copy()
d = d[pd.notna(d[elapsed_col])].copy()

# binary: did home score in B1?
d["home_scored_b1"] = (d["home_runs_bottom1"].fillna(0) > 0).astype(int)

# --- Bin elapsed seconds (quintiles)
# guard against ties by adding tiny noise for qcut
_eps = np.random.default_rng(0).normal(0, 1e-6, len(d))
d["_elapsed_jitter"] = d[elapsed_col] + _eps
d["elapsed_bin"] = pd.qcut(d["_elapsed_jitter"], 5, labels=[1, 2, 3, 4, 5])
d.drop(columns=["_elapsed_jitter"], inplace=True)

# --- Aggregate overall
overall = (
    d.groupby("elapsed_bin", observed=True)
     .agg(
         games=("home_scored_b1", "size"),
         pct_scored=("home_scored_b1", "mean"),
         avg_elapse_sec=(elapsed_col, "mean"),
     )
     .reset_index()
)

overall["pct_scored"] = overall["pct_scored"].astype(float)
overall.to_csv(OUT_CSV, index=False)

# --- (Optional) By season, for robustness
by_year = (
    d.groupby(["season", "elapsed_bin"], observed=True)
     .agg(
         games=("home_scored_b1", "size"),
         pct_scored=("home_scored_b1", "mean"),
     )
     .reset_index()
)

# --- Console output
print("Rows used:", len(d))
print("\nOverall (quintiles shortest → longest):")
print(overall)
print(f"\nWrote: {OUT_CSV}")
