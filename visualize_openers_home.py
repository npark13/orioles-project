import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load aggregated results ---
file_path = "/Users/kevinhe/orioles-project/data/out/team_home_field_by_year/first_inning_nb_results_home_2018_2024.csv"
df = pd.read_csv(file_path)

# --- Sort by home_advantage ---
df = df.sort_values("home_advantage", ascending=False).reset_index(drop=True)

# --- Create bar chart ---
plt.figure(figsize=(12, 6))
colors = ["red" if team == "ANA" else "steelblue" for team in df["team"]]  # Highlight Angels (ANA)

sns.barplot(x="team", y="home_advantage", data=df, palette=colors)

# --- Labels and title ---
plt.xticks(rotation=45, ha='right')
plt.ylabel("First-Inning Home Advantage")
plt.xlabel("Team")
plt.title("MLB Teams: Aggregated First-Inning Home Advantage (as home team) (2018–2024)\nLos Angeles Angels Highlighted in Red")

# --- Add values on top of bars ---
for index, row in df.iterrows():
    plt.text(index, row.home_advantage + 0.01, f"{row.home_advantage:.2f}", ha='center', va='bottom', fontsize=8)

plt.tight_layout()

# --- Save figure ---
output_file = "/Users/kevinhe/orioles-project/data/out/team_home_field_by_year/first_inning_home_advantagehome_2018_2024.png"
plt.savefig(output_file, dpi=300)
print(f"✅ Figure saved to {output_file}")

plt.show()