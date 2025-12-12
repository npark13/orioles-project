import pandas as pd
import matplotlib.pyplot as plt

# Load the data
summary = pd.read_csv("/Users/kevinhe/orioles-project/first_inning_nb_summary.csv")
basic = pd.read_csv("/Users/kevinhe/orioles-project/data/out/first_inning_nb_results.csv")
travel = pd.read_csv("/Users/kevinhe/orioles-project/data/out/first_inning_nb_results_with_travel.csv")

# Merge on year
df = summary.merge(
    basic[['year','home_advantage']], on='year', how='left', suffixes=('_no_travel','_basic')
).merge(
    travel[['year','home_advantage']], on='year', how='left'
)
df.rename(columns={
    'home_advantage_no_travel':'home_advantage_only',
    'home_advantage_basic':'home_advantage_era',
    'home_advantage':'home_advantage_era_travel'
}, inplace=True)

# Filter 2014-2024 safely
df_plot = df.loc[(df['year'] >= 2014) & (df['year'] <= 2024)].copy()

# Compute differences
df_plot['era_diff'] = df_plot['home_advantage_era'] - df_plot['home_advantage_only']
df_plot['era_travel_diff'] = df_plot['home_advantage_era_travel'] - df_plot['home_advantage_only']

# ---------------------
# Combined figure with 2 subplots
# ---------------------
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,10), sharex=True)

# --- Top plot: full home advantage ---
axes[0].plot(df_plot['year'], df_plot['home_advantage_only'], marker='o', label='Just Home Advantage')
axes[0].plot(df_plot['year'], df_plot['home_advantage_era'], marker='s', label='Home Advantage + ERA')
axes[0].plot(df_plot['year'], df_plot['home_advantage_era_travel'], marker='^', label='Home Advantage + ERA + Travel')
axes[0].set_ylabel("Estimated Home Advantage (Runs)")
axes[0].set_title("MLB First-Inning Home Advantage: 2014-2024")
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend()

# --- Bottom plot: differences from baseline ---
axes[1].plot(df_plot['year'], [0]*len(df_plot), marker='o', linestyle='--', label='Baseline: Just Home Advantage')
axes[1].plot(df_plot['year'], df_plot['era_diff'], marker='s', label='ERA Adjusted Difference')
axes[1].plot(df_plot['year'], df_plot['era_travel_diff'], marker='^', label='ERA + Travel Adjusted Difference')
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Difference from Home Advantage Only (Runs)")
axes[1].set_title("MLB First-Inning Home Advantage Differences: 2014-2024")
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend()

plt.xticks(df_plot['year'], rotation=45)
plt.tight_layout()

# Save combined figure
plt.savefig("/Users/kevinhe/orioles-project/home_advantage_combined.png", dpi=300)
plt.close()

print("Combined plot saved to /Users/kevinhe/orioles-project/home_advantage_combined.png")