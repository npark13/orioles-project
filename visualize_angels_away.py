import pandas as pd
import matplotlib.pyplot as plt
import glob

# --- Load data ---
files = sorted(glob.glob("/Users/kevinhe/orioles-project/data/out/team_home_field_by_year/first_inning_nb_results_vis_*.csv"))
df_all = pd.concat([pd.read_csv(f) for f in files])

# --- Compute league average per year ---
league_avg = df_all.groupby("year", as_index=False)["home_advantage"].mean()
league_avg["team"] = "LEAGUE_AVG"

# --- Figure 1: Year-by-Year Trends ---
plt.figure(figsize=(12, 7))

# Plot all teams (light gray)
for team, team_df in df_all.groupby("team"):
    if team != "ANA":
        plt.plot(team_df["year"], team_df["home_advantage"],
                 color="lightgray", linewidth=1, alpha=0.7)

# Highlight Angels
angels_df = df_all[df_all["team"] == "ANA"]
plt.plot(angels_df["year"], angels_df["home_advantage"],
         color="red", linewidth=3, label="Los Angeles Angels")

# League average line
plt.plot(league_avg["year"], league_avg["home_advantage"],
         color="black", linestyle="--", linewidth=2, label="League Average")

plt.title("First-Inning Home Field Advantage (as Away Team) (2018–2024)\nAngels vs. League", fontsize=16, weight='bold')
plt.xlabel("Year")
plt.ylabel("Home Advantage (Predicted Runs Difference)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/Users/kevinhe/orioles-project/data/out/angels_vs_league_trends_away.png", dpi=300)
plt.show()

# --- Figure 2: Average Home Advantage Across All Years ---
avg_advantage = df_all.groupby("team", as_index=False)["home_advantage"].mean()
avg_advantage = avg_advantage.sort_values("home_advantage", ascending=False)

# Compute league-wide mean
league_mean = avg_advantage["home_advantage"].mean()

plt.figure(figsize=(12, 7))
plt.bar(avg_advantage["team"], avg_advantage["home_advantage"],
        color=["red" if t == "ANA" else "lightgray" for t in avg_advantage["team"]])

# League average line
plt.axhline(y=league_mean, color="black", linestyle="--", linewidth=2, label="League Average")

plt.title("Average First-Inning Home Field Advantage (as Away Team) (2018–2024)", fontsize=16, weight='bold')
plt.ylabel("Average Home Advantage")
plt.xticks(rotation=90)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("/Users/kevinhe/orioles-project/data/out/angels_vs_league_average_away.png", dpi=300)
plt.show()