import pandas as pd
import matplotlib.pyplot as plt

# --- Load the data ---
mlb_travel = pd.read_csv("/Users/kevinhe/orioles-project/data/out/first_inning_nb_results_with_travel.csv")
angels_travel = pd.read_csv("/Users/kevinhe/orioles-project/data/out/first_inning_nb_results_with_travel_angels.csv")

# --- Merge on year for safe comparison ---
df = mlb_travel[['year','home_advantage']].rename(columns={'home_advantage':'home_advantage_mlb'}).merge(
    angels_travel[['year','home_advantage']].rename(columns={'home_advantage':'home_advantage_angels'}),
    on='year', how='left'
)

# Filter 2014-2024 safely
df_plot = df.loc[(df['year'] >= 2018) & (df['year'] <= 2024)].copy()

# Compute differences from MLB baseline
df_plot['angels_diff'] = df_plot['home_advantage_angels'] - df_plot['home_advantage_mlb']

# ---------------------
# Combined figure with 2 subplots
# ---------------------
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,10), sharex=True)

# --- Top plot: full home advantage ---
axes[0].plot(df_plot['year'], df_plot['home_advantage_mlb'], marker='o', label='MLB First-Inning Home Advantage')
axes[0].plot(df_plot['year'], df_plot['home_advantage_angels'], marker='^', label='Anaheim Angels Only')
axes[0].set_ylabel("Estimated Home Advantage (Runs)")
axes[0].set_title("MLB vs Anaheim Angels: First-Inning Home Advantage (2014-2024)")
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend()

# --- Bottom plot: differences from MLB baseline ---
axes[1].plot(df_plot['year'], [0]*len(df_plot), marker='o', linestyle='--', label='MLB Baseline')
axes[1].plot(df_plot['year'], df_plot['angels_diff'], marker='^', label='Angels vs MLB Difference')
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Difference from MLB Home Advantage (Runs)")
axes[1].set_title("Anaheim Angels Difference from MLB First-Inning Home Advantage: 2014-2024")
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend()

plt.xticks(df_plot['year'], rotation=45)
plt.tight_layout()

# --- Save combined figure ---
plt.savefig("/Users/kevinhe/orioles-project/home_advantage_mlb_vs_angels.png", dpi=300)
plt.close()

print("Combined plot saved to /Users/kevinhe/orioles-project/home_advantage_mlb_vs_angels.png")
