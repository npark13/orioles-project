import pandas as pd
import matplotlib.pyplot as plt

# --- Load data ---
mlb = pd.read_csv("/Users/kevinhe/orioles-project/data/out/first_inning_nb_results_with_travel.csv")
angels = pd.read_csv("/Users/kevinhe/orioles-project/data/out/first_inning_nb_results_with_travel_angels_combined.csv")

# --- Filter to relevant years and subsets ---
mlb = mlb[(mlb['year'] >= 2018) & (mlb['year'] <= 2024)]
angels_home = angels[(angels['year'] >= 2018) & (angels['year'] <= 2024) & (angels['subset'] == 'home')]

# --- Compute averages for reference lines ---
mlb_avg = mlb['home_advantage'].mean()
angels_home_avg = angels_home['home_advantage'].mean()

# --- Compute MLB standard deviation band ---
mlb_std = mlb['home_advantage'].std()
mlb_upper = mlb['home_advantage'] + mlb_std
mlb_lower = mlb['home_advantage'] - mlb_std

# --- Create clean plot ---
plt.figure(figsize=(9,6))

# Shaded MLB ±1σ band
plt.fill_between(mlb['year'], mlb_lower, mlb_upper, color='lightgray', alpha=0.4, label='MLB ±1 SD')

# MLB average line
plt.plot(mlb['year'], mlb['home_advantage'], color='black', marker='o', linewidth=2.5, label='MLB Average')

# Angels HOME line
plt.plot(angels_home['year'], angels_home['home_advantage'], color='red', marker='s', linewidth=2.5, label='Angels Home')

# Horizontal averages
plt.axhline(mlb_avg, color='black', linestyle='--', linewidth=1.2, label=f'MLB Mean = {mlb_avg:.3f}')
plt.axhline(angels_home_avg, color='red', linestyle='--', linewidth=1.2, label=f'Angels Home Mean = {angels_home_avg:.3f}')

# Labels and aesthetics
plt.title("First-Inning Home-Field Advantage (2018–2024)\nAngels Home vs. Rest of MLB", fontsize=14, weight='bold')
plt.xlabel("Year", fontsize=12)
plt.ylabel("Expected Runs (Home Advantage)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=False, fontsize=10)
plt.xticks(mlb['year'])
plt.tight_layout()

# Save
plt.savefig("/Users/kevinhe/orioles-project/angels_home_vs_mlb_clean.png", dpi=300)
plt.close()

print("✅ Clean Angels home vs. MLB comparison chart saved to /Users/kevinhe/orioles-project/angels_home_vs_mlb_clean.png")