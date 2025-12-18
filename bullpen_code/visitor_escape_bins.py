import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SRC = "csv_files/binned_first_inning_stats.csv"
OUT_CSV = "csv_files/binned_first_inning_stats_with_escape.csv"
OUT_PNG = Path("all_visuals/bullpen_cooldown_visuals") / "visitor_escape_bins.png"
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

# 1) Load
df = pd.read_csv(SRC)

# 2) Add visitor escape rate = 1 - (home scored %)
df["visitor_escape_rate"] = 1 - df["pct_scored"]

# 3) Save the augmented table
df.to_csv(OUT_CSV, index=False)

# 4) Plot
plt.figure(figsize=(8, 5))
plt.plot(df["elapsed_bin"], df["visitor_escape_rate"]*100, marker="D", linestyle="-")

# annotate each point with the average elapsed seconds label (e.g., "197s")
for _, r in df.iterrows():
    plt.text(r["elapsed_bin"], r["visitor_escape_rate"]*100 + 0.5, f'{int(round(r["avg_elapse_sec"]))}s',
             ha="center", fontsize=9)

plt.title("Visitor Pitcher Escape Rate vs Top-1st Elapsed Time (2015–present)")
plt.xlabel("Top-1st Elapsed Time Quintile (1=shortest, 5=longest)")
plt.ylabel("Visitor Escape w/ 0 Runs Allowed in Bottom-1 (%)")
plt.ylim(80, 100)                 # clean % scale; tweak if your values differ
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
print(f"[OK] Wrote {OUT_CSV} and {OUT_PNG}")
