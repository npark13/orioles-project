import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def plot_runs_per_inning(df, save_path="runs_per_inning.png"):
    # Drop rows w/o the needed data
    df = df.dropna(subset=["inning", "visitor_avg_runs", "home_avg_runs", "games"])
    
    # Weight averages by games per year
    summary = []
    for inning in sorted(df["inning"].unique()):
        d = df[df["inning"] == inning]
        visitor_total = (d["visitor_avg_runs"] * d["games"]).sum()
        home_total = (d["home_avg_runs"] * d["games"]).sum()
        total_games = d["games"].sum()
        summary.append({
            "inning": inning,
            "visitor_avg": visitor_total / total_games,
            "home_avg": home_total / total_games,
        })
    
    # Turn our summary into a DataFrame and add a total column
    df_plot = pd.DataFrame(summary)
    df_plot["total_avg_runs"] = df_plot["visitor_avg"] + df_plot["home_avg"]

    # Plot total average runs per inning
    plt.figure(figsize=(10, 6))
    plt.plot(df_plot["inning"], df_plot["total_avg_runs"], linestyle="-", marker='o', color="black", label="Total")
    plt.xlabel("Inning")
    plt.ylabel("Average Runs per Inning")
    plt.title("Average Runs per Inning (Both Teams, 2013-2024)")
    plt.xticks(range(1, 11))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def home_versus_visiting_inning(df, save_path="home_versus_visiting_inning.png"):
    # Drop rows w/o the needed data
    df = df.dropna(subset=["inning", "visitor_avg_runs", "home_avg_runs", "games"])
    
    # Compute weighted average runs for home vs visiting teams
    summary = []
    for inning in sorted(df["inning"].unique()):
        d = df[df["inning"] == inning]
        visitor_total = (d["visitor_avg_runs"] * d["games"]).sum()
        home_total = (d["home_avg_runs"] * d["games"]).sum()
        total_games = d["games"].sum()
        summary.append({
            "inning": inning,
            "visitor_avg": visitor_total / total_games,
            "home_avg": home_total / total_games,
        })
    
    df_plot = pd.DataFrame(summary)

    # Plot home vs visiting runs per inning
    plt.figure(figsize=(10, 6))
    plt.plot(df_plot["inning"], df_plot["visitor_avg"], linestyle="-", marker='o', color="blue", label="Visitor")
    plt.plot(df_plot["inning"], df_plot["home_avg"], linestyle="-", marker='o', color="red", label="Home")
    plt.xlabel("Inning")
    plt.ylabel("Runs per Inning")
    plt.title("Average Runs per Inning by Visitor and Home, 2013-2024")
    plt.xticks(range(1, 11))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_differential_runs_per_inning(df, save_path="differential_runs_per_inning.png"):
    # Drop rows w/o the needed data
    df = df.dropna(subset=["inning", "visitor_avg_runs", "home_avg_runs", "games"])
    
    # Calculate average difference in runs per inning
    summary = []
    for inning in sorted(df["inning"].unique()):
        d = df[df["inning"] == inning]
        visitor_total = (d["visitor_avg_runs"] * d["games"]).sum()
        home_total = (d["home_avg_runs"] * d["games"]).sum()
        total_games = d["games"].sum()
        summary.append({
            "inning": inning,
            "diff_avg": (home_total - visitor_total) / total_games,
        })
    
    df_plot = pd.DataFrame(summary)

    # Plot home minus visitor differential
    plt.figure(figsize=(10, 6))
    plt.plot(df_plot["inning"], df_plot["diff_avg"], linestyle="-", marker='o', color="black", label="Home - Visitor")
    plt.xlabel("Inning")
    plt.ylabel("Run Differential")
    plt.title("Run Differential by Inning (Home minus Visitor, 2013-2024)")
    plt.xticks(range(1, 11))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_differential_runs_first_inning(df, save_path="differential_runs_first_inning.png"):
    if "home_minus_vis_runs" not in df.columns or "year" not in df.columns:
        print("Skipping plot_differential_runs_first_inning: required columns missing.")
        return

    df = df.dropna(subset=["home_minus_vis_runs", "year"])

    has_games = "games" in df.columns

    # Only needed columns
    value_cols = ["home_minus_vis_runs"]
    if has_games:
        value_cols.append("games")

    # --- FIX: Build a clean frame and group on year separately ---
    df_group = df[["year"]]             # grouping column only
    df_values = df[value_cols]          # NO GROUPING COLUMN AT ALL

    grouped = df_group.groupby("year")  # group only the year index

    # Weighted average using df_values rows
    def weighted_avg(idxs):
        d = df_values.loc[idxs.index]
        values = d["home_minus_vis_runs"]
        weights = d["games"] if has_games else 1
        return (values * weights).sum() / weights.sum()

    summary = grouped.apply(weighted_avg).reset_index(name="diff")

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(summary["year"], summary["diff"], linestyle="-", marker='o', color="black")
    plt.xlabel("Year")
    plt.ylabel("Run Differential")
    plt.title("First Inning Run Differential by Year")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_home_vs_visitor_first_inning_line(df, save_path="home_vs_visitor_first_inning.png"):
    # Find the right columns for x (visitor) and y (home first-inning)
    col_x = next((c for c in df.columns if "visitor" in c.lower()), None)
    col_y = next((c for c in df.columns if "home" in c.lower() and "first" in c.lower()), None)

    if col_x is None or col_y is None:
        print("Skipping plot_home_vs_visitor_first_inning_line: required columns missing.")
        return

    # Plot the correlation
    plt.figure(figsize=(8, 5))
    plt.plot(df[col_x], df[col_y], linestyle="-", marker="o")
    plt.xlabel("Visitor Runs in First Inning")
    plt.ylabel("Average Home Team Runs in First Inning")
    plt.title("Correlation of Home Scoring and Visitor Scoring")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_first_inning_diff_by_decade(df, save_path="first_inning_diff_by_decade.png"):
    """
    Aggregates per-year first-inning stats by decade using games as weights,
    then plots:
    - Average First Inning Run Differential (Home - Visitor) by decade
    """
    # Compute decade for each year
    df['decade'] = (df['year'] // 10) * 10
    decades = sorted(df['decade'].unique())
    
    summary = []
    for dec in decades:
        d = df[df['decade'] == dec]
        total_games = d['games'].sum()

        # Weighted average of run differential
        diff_avg = ((d['home_avg_runs_1st'] - d['visitor_avg_runs_1st']) * d['games']).sum() / total_games

        summary.append({
            'decade': dec,
            'diff_avg': diff_avg
        })

    df_plot = pd.DataFrame(summary)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_plot['decade'].astype(str) + 's', df_plot['diff_avg'], marker='o', color='black')
    plt.xlabel("Decade")
    plt.ylabel("Average First Inning Run Differential (Home - Visitor)")
    plt.title("First Inning Run Differential by Decade (Home - Visitor)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_first_inning_summary_by_decade(df, save_path="first_inning_summary_by_decade.png"):
    """
    Aggregates per-year first-inning stats by decade using games as weights,
    then plots:
    - Expected runs for home vs away teams
    - Home advantage
    """
    # Compute decade for each year
    df['decade'] = (df['year'] // 10) * 10

    decades = sorted(df['decade'].unique())
    summary = []

    for dec in decades:
        d = df[df['decade'] == dec]
        total_games = d['games'].sum()

        # Weighted averages like plot_runs_per_inning
        away_avg = (d['visitor_avg_runs_1st'] * d['games']).sum() / total_games
        home_avg = (d['home_avg_runs_1st'] * d['games']).sum() / total_games
        home_adv = home_avg / away_avg

        summary.append({
            'decade': dec,
            'E[Y_away]': away_avg,
            'E[Y_home]': home_avg,
            'home_advantage': home_adv
        })

    df_plot = pd.DataFrame(summary)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 15), sharex=True)
    axes[0].plot(df_plot['decade'].astype(str), df_plot['E[Y_away]'], marker='o', label='Away')
    axes[0].plot(df_plot['decade'].astype(str), df_plot['E[Y_home]'], marker='o', label='Home')
    axes[0].set_ylabel("Expected First-Inning Runs")
    axes[0].set_title("Expected First-Inning Runs by Home/Away Team Over Decades")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df_plot['decade'].astype(str), df_plot['home_advantage'], marker='o', color='green')
    axes[1].set_ylabel("Home Advantage (E[Y_home] / E[Y_away])")
    axes[1].set_title("Home Advantage in First-Inning Runs Over Decades")
    axes[1].set_xlabel("Decade")
    axes[1].grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="all_visuals/retrosheet_remake_visuals")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = "data/out/inning_summary.csv"
    csv_path_two = "data/out/first_inning_summary.csv"
    csv_path_three = "data/out/visitor_vs_home_first_inning.csv"


    # Load the data
    df = pd.read_csv(csv_path)      # inning-level data
    df_2 = pd.read_csv(csv_path_two)  # first-inning per-year data
    df_3 = pd.read_csv(csv_path_three)  # home vs visitor scatter

    # --- Filter for 2013–2024 only for inning-level plots ---
    if "year" in df.columns:
        df = df[(df["year"] >= 2013) & (df["year"] <= 2024)]

    # --- Plots ---

    # 1. First Inning Differential by Decade (1910s–2020s)
    plot_first_inning_diff_by_decade(df_2)  

    # 2. Inning-level plots (2013–2024)
    plot_runs_per_inning(df)
    home_versus_visiting_inning(df)
    plot_differential_runs_per_inning(df)

    # 3. First-inning differential per year (2013–2024)
    plot_differential_runs_first_inning(df_2)

    # 4. Scatter: home vs visitor first-inning
    plot_home_vs_visitor_first_inning_line(df_3)

if __name__ == "__main__":
    main()