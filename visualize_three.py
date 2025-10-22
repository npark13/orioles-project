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

# ---------------------
# Original plot (3 lines)
# ---------------------
plt.figure(figsize=(10,6))
plt.plot(df_plot['year'], df_plot['home_advantage_only'], marker='o', label='Just Home Advantage')
plt.plot(df_plot['year'], df_plot['home_advantage_era'], marker='s', label='Home Advantage + ERA')
plt.plot(df_plot['year'], df_plot['home_advantage_era_travel'], marker='^', label='Home Advantage + ERA + Travel')

plt.xlabel("Year")
plt.ylabel("Estimated Home Advantage (Runs)")
plt.title("MLB First-Inning Home Advantage: 2014-2024")
plt.xticks(df_plot['year'], rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

# Save the first plot
plt.savefig("/Users/kevinhe/orioles-project/home_advantage_full.png", dpi=300)
plt.close()

# ---------------------
# Difference plot (baseline at 0)
# ---------------------
df_plot['era_diff'] = df_plot['home_advantage_era'] - df_plot['home_advantage_only']
df_plot['era_travel_diff'] = df_plot['home_advantage_era_travel'] - df_plot['home_advantage_only']

plt.figure(figsize=(10,6))
plt.plot(df_plot['year'], [0]*len(df_plot), marker='o', linestyle='--', label='Just Home Advantage (Baseline)')
plt.plot(df_plot['year'], df_plot['era_diff'], marker='s', label='ERA Adjusted Difference')
plt.plot(df_plot['year'], df_plot['era_travel_diff'], marker='^', label='ERA + Travel Adjusted Difference')

plt.xlabel("Year")
plt.ylabel("Difference from Home Advantage Only (Runs)")
plt.title("MLB First-Inning Home Advantage Differences: 2014-2024")
plt.xticks(df_plot['year'], rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

# Save the second plot
plt.savefig("/Users/kevinhe/orioles-project/home_advantage_diff.png", dpi=300)
plt.close()

print("Plots saved to /Users/kevinhe/orioles-project/")