import pandas as pd
import matplotlib.pyplot as plt

# --- File paths ---
summary_file = "/Users/kevinhe/orioles-project/data/out/first_inning_nb_results.csv"
travel_file = "/Users/kevinhe/orioles-project/data/out/first_inning_nb_results_with_travel.csv"
summary_file_openers = "/Users/kevinhe/orioles-project/data/out/first_inning_nb_results_openers.csv"
travel_file_openers = "/Users/kevinhe/orioles-project/data/out/first_inning_nb_results_with_travel_openers.csv"

# --- Load the data ---
summary = pd.read_csv(summary_file)
travel = pd.read_csv(travel_file)
summary_openers = pd.read_csv(summary_file_openers)
travel_openers = pd.read_csv(travel_file_openers)

# --- Rename travel columns ---
travel = travel.rename(columns={'home_advantage':'home_advantage_travel'})
travel_openers = travel_openers.rename(columns={'home_advantage':'home_advantage_travel'})

# --- Merge ---
df_all = summary.merge(travel[['year','home_advantage_travel']], on='year', how='left')
df_openers = summary_openers.merge(travel_openers[['year','home_advantage_travel']], on='year', how='left')

# --- Compute travel difference ---
df_all['travel_diff'] = df_all['home_advantage_travel'] - df_all['home_advantage']
df_openers['travel_diff'] = df_openers['home_advantage_travel'] - df_openers['home_advantage']

# --- Filter years ---
years = list(range(2014, 2025))
df_all = df_all[df_all['year'].isin(years)]
df_openers = df_openers[df_openers['year'].isin(years)]

# --- Save travel difference CSVs ---
df_all[['year','travel_diff']].to_csv("/Users/kevinhe/orioles-project/data/out/home_advantage_travel_diff.csv", index=False)
df_openers[['year','travel_diff']].to_csv("/Users/kevinhe/orioles-project/data/out/home_advantage_travel_diff_openers.csv", index=False)
print("Difference CSVs saved.")

# ---------------------
# Figure 1: All Games
# ---------------------
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,8), sharex=True)

axes[0].plot(df_all['year'], df_all['home_advantage'], marker='o', label='Just Home Advantage')
axes[0].plot(df_all['year'], df_all['home_advantage_travel'], marker='^', label='Home Advantage + Travel')
axes[0].set_ylabel("Estimated Home Advantage (Runs)")
axes[0].set_title("MLB First-Inning Home Advantage: All Games 2014-2024")
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend()

axes[1].plot(df_all['year'], [0]*len(df_all), marker='o', linestyle='--', label='Baseline: Just Home Advantage')
axes[1].plot(df_all['year'], df_all['travel_diff'], marker='^', label='Travel Adjusted Difference')
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Difference from Home Advantage Only (Runs)")
axes[1].set_title("MLB First-Inning Home Advantage Differences: All Games 2014-2024")
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend()

plt.xticks(years, rotation=45)
plt.tight_layout()
plt.savefig("/Users/kevinhe/orioles-project/home_advantage_travel_all_games.png", dpi=300)
plt.close()
print("Figure for all games saved.")

# ---------------------
# Figure 2: Series Openers
# ---------------------
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,8), sharex=True)

axes[0].plot(df_openers['year'], df_openers['home_advantage'], marker='o', label='Just Home Advantage')
axes[0].plot(df_openers['year'], df_openers['home_advantage_travel'], marker='^', label='Home Advantage + Travel')
axes[0].set_ylabel("Estimated Home Advantage (Runs)")
axes[0].set_title("MLB First-Inning Home Advantage: Series Openers 2014-2024")
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend()

axes[1].plot(df_openers['year'], [0]*len(df_openers), marker='o', linestyle='--', label='Baseline: Just Home Advantage')
axes[1].plot(df_openers['year'], df_openers['travel_diff'], marker='^', label='Travel Adjusted Difference')
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Difference from Home Advantage Only (Runs)")
axes[1].set_title("MLB First-Inning Home Advantage Differences: Series Openers 2014-2024")
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend()

plt.xticks(years, rotation=45)
plt.tight_layout()
plt.savefig("/Users/kevinhe/orioles-project/home_advantage_travel_openers.png", dpi=300)
plt.close()
print("Figure for series openers saved.")