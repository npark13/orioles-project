#!/usr/bin/env python3
# travel_code/plot_angels_dir_compare.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN  = PROJECT_ROOT / "angels_dir_compare.csv"
OUT = PROJECT_ROOT / "angels_dir_compare.png"

if not IN.exists():
    raise SystemExit(f"Missing {IN}. Run winning_vs_travel_2013_2024.py first.")

df = pd.read_csv(IN)
# Map 0/1 to labels
df["group"] = np.where(df["visitor_is_angels"]==1, "Angels", "League (non-Angels)")

# Keep a consistent x order
order = ["eastbound", "same_zone", "westbound"]
df["dir_label"] = pd.Categorical(df["dir_label"], categories=order, ordered=True)

# Pivot to wide: rows = dir_label, cols = group
wide = df.pivot(index="dir_label", columns="group", values="home_win_pct").reindex(order)

# Plot grouped bars
fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
x = np.arange(len(wide.index))
width = 0.36

ax.bar(x - width/2, wide["League (non-Angels)"].values, width, label="League (non-Angels)")
ax.bar(x + width/2, wide["Angels"].values, width, label="Angels")

# Labels/formatting
ax.set_xticks(x)
ax.set_xticklabels(wide.index)
ax.set_ylabel("Home Winning Percentage")
ax.set_title("Angels vs League by Visitor Time-Zone Direction (2013–2024, Series Openers)")
ax.grid(True, axis="y", alpha=0.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="best")

# Annotate bars with percentages
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
