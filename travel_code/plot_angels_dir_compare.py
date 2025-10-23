#!/usr/bin/env python3
# -------------------------------------------------------------
# plot_angels_dir_compare.py (fixed column names)
# -------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN  = PROJECT_ROOT / "angels_dir_compare.csv"
OUT = PROJECT_ROOT / "angels_dir_compare.png"

if not IN.exists():
    raise SystemExit("Missing angels_dir_compare.csv — run winning_vs_travel_2013_2024.py first.")

df = pd.read_csv(IN)

# Map 0/1 to readable group names
df["group"] = np.where(df["visitor_is_angels"]==1, "Angels", "League (non-Angels)")

order = ["eastbound", "same_zone", "westbound"]
df["dir_label"] = pd.Categorical(df["dir_label"], categories=order, ordered=True)

# Pivot, rename integer columns if needed
wide = df.pivot(index="dir_label", columns="group", values="home_win_pct").reindex(order)

# Handle both numeric or string column cases
if "Angels" not in wide.columns:
    cols = wide.columns.to_list()
    rename_map = {}
    if 1 in cols: rename_map[1] = "Angels"
    if 0 in cols: rename_map[0] = "League (non-Angels)"
    wide = wide.rename(columns=rename_map)

fig, ax = plt.subplots(figsize=(7.5,4.5), dpi=160)
x = np.arange(len(wide.index))
width = 0.36

ax.bar(x - width/2, wide["League (non-Angels)"], width, label="League (non-Angels)")
ax.bar(x + width/2, wide["Angels"], width, label="Angels")

# Labels & formatting
ax.set_xticks(x)
ax.set_xticklabels(wide.index)
ax.set_ylabel("Home Winning Percentage")
ax.set_title("Angels vs League by Visitor Travel Direction (Significant Travel Only, 2013–2024)")
ax.grid(True, axis="y", alpha=0.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="best")

# Annotate bars
for bars in ax.containers:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

fig.tight_layout()
fig.savefig(OUT)
print(f"[OK] Saved {OUT}")
