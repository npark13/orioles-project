import pandas as pd
import matplotlib.pyplot as plt

SRC = "binned_first_inning_stats.csv"          # your existing file
OUT_PNG = "home_vs_visitor_first_inning_bins.png"
OUT_CSV = "binned_first_inning_stats_with_escape.csv"

# 1) Load
df = pd.read_csv(SRC)

# 2) Visitor escape rate = 1 - home scored %
df["visitor_escape_rate"] = 1 - df["pct_scored"]

# Save an augmented copy (handy for slides / audit)
df.to_csv(OUT_CSV, index=False)

# 3) Plot both series on the same figure (dual y-axes for clarity)
fig, ax1 = plt.subplots(figsize=(8.5, 5.2), dpi=200)

# Left axis: Home scored %
ax1.plot(df["elapsed_bin"], df["pct_scored"]*100, marker="D", linestyle="-", label="Home scored in B1 (%)")
# Annotate with avg elapsed seconds under each point
for _, r in df.iterrows():
    ax1.text(r["elapsed_bin"], r["pct_scored"]*100 + 0.4, f'{int(round(r["avg_elapse_sec"]))}s',
             ha="center", va="bottom", fontsize=9)

ax1.set_xlabel("Top-1st Elapsed Time Quintile (1=shortest, 5=longest)")
ax1.set_ylabel("Home scored in Bottom-1st (%)")
ax1.grid(True, axis="y", linestyle="--", alpha=0.7)

# Right axis: Visitor escape %
ax2 = ax1.twinx()
ax2.plot(df["elapsed_bin"], df["visitor_escape_rate"]*100, marker="D", linestyle="--", label="Visitor escaped B1 (0 runs, %)")
ax2.set_ylabel("Visitor escaped Bottom-1st (%)")

# Title + legend
plt.title("First-Inning Outcomes vs Top-1st Elapsed Time (2015–present)")
# Build a combined legend from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="upper left")

plt.tight_layout()
plt.savefig(OUT_PNG)
print(f"[OK] Wrote {OUT_CSV} and {OUT_PNG}")
