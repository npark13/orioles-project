import pandas as pd
import matplotlib.pyplot as plt

def plot_runs_per_inning(df, save_path="runs_per_inning.png"):
    df = df.dropna(subset=["inning", "visitor_avg_runs", "home_avg_runs"])
    df["total_avg_runs"] = df["visitor_avg_runs"] + df["home_avg_runs"]

    plt.figure(figsize=(10, 6))
    plt.plot(df["inning"], df["total_avg_runs"], linestyle="-", marker='o', color="black", label="Total")
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
    df = df.dropna(subset=["inning", "visitor_avg_runs", "home_avg_runs"])

    plt.figure(figsize=(10, 6))
    plt.plot(df["inning"], df["visitor_avg_runs"], linestyle="-", marker='o', color="blue", label="Visitor")
    plt.plot(df["inning"], df["home_avg_runs"], linestyle="-", marker='o', color="red", label="Home")
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
    df = df.dropna(subset=["inning", "visitor_avg_runs", "home_avg_runs"])
    df["differential_avg_runs"] = df["home_avg_runs"] - df["visitor_avg_runs"]

    plt.figure(figsize=(10, 6))
    plt.plot(df["inning"], df["differential_avg_runs"], linestyle="-", marker='o', color="black", label="Home - Visitor")
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

    plt.figure(figsize=(10, 6))
    plt.plot(df["year"], df["home_minus_vis_runs"], linestyle="-", marker='o', color="black")
    plt.xlabel("Year")
    plt.ylabel("Run Differential")
    plt.title("First Inning Run Differential by Year")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_home_vs_visitor_first_inning_line(df, save_path="home_vs_visitor_first_inning.png"):
    # Check for expected columns
    col_x = next((c for c in df.columns if "visitor" in c.lower()), None)
    col_y = next((c for c in df.columns if "home" in c.lower() and "first" in c.lower()), None)

    if col_x is None or col_y is None:
        print("Skipping plot_home_vs_visitor_first_inning_line: required columns missing.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(df[col_x], df[col_y], linestyle="-", marker="o")
    plt.xlabel("Visitor Runs in First Inning")
    plt.ylabel("Average Home Team Runs in First Inning")
    plt.title("Correlation of Home Scoring and Visitor Scoring")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_average_travel_bar(save_path="average_miles_traveled_bar.png"):
    # Load the CSV
    travel_csv = "/Users/kevinhe/orioles-project/team_travel_stats_with_teams.csv"
    df_travel = pd.read_csv(travel_csv)
    
    # Ensure numeric type for Average column
    df_travel["Average"] = pd.to_numeric(df_travel["Average"], errors="coerce")
    df_travel = df_travel.dropna(subset=["Average", "Year"])
    
    plt.figure(figsize=(10, 6))
    plt.bar(df_travel["Year"], df_travel["Average"], color="cornflowerblue", alpha=0.7)
    plt.xlabel("Year")
    plt.ylabel("Average Miles Traveled")
    plt.title("Average Miles Traveled per Team (2014-2024)")
    plt.xticks(df_travel["Year"])  # Show all years on x-axis
    plt.yticks([10000, 20000, 30000, 40000])  # Set y-axis ticks manually
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_first_inning_summary_by_decade(csv_path, save_path="first_inning_summary_by_decade.png"):
    """
    Reads a decade-aggregated CSV with columns:
    'decade', 'E[Y_away]', 'E[Y_home]', 'alpha', 'home_advantage'
    and plots expected runs, home advantage, and dispersion (alpha)
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values('decade')
    decades = df['decade'].astype(str)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Expected runs home vs away
    axes[0].plot(decades, df['E[Y_away]'], marker='o', label='Away')
    axes[0].plot(decades, df['E[Y_home]'], marker='o', label='Home')
    axes[0].set_ylabel("Expected First-Inning Runs")
    axes[0].set_title("Expected First-Inning Runs by Home/Away Team Over Decades")
    axes[0].legend()
    axes[0].grid(True)
    
    # Home advantage
    axes[1].plot(decades, df['home_advantage'], marker='o', color='green')
    axes[1].set_ylabel("Home Advantage (E[Y_home] / E[Y_away])")
    axes[1].set_title("Home Advantage in First-Inning Runs Over Decades")
    axes[1].grid(True)
    
    # Dispersion (alpha)
    axes[2].plot(decades, df['alpha'], marker='o', color='red')
    axes[2].set_ylabel("Dispersion Parameter α")
    axes[2].set_title("Overdispersion in First-Inning Runs Over Decades")
    axes[2].set_xlabel("Decade")
    axes[2].grid(True)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Load CSVs
    csv_path = "/Users/kevinhe/orioles-project/data/out/inning_summary.csv"
    csv_path_two = "/Users/kevinhe/orioles-project/data/out/first_inning_summary.csv"
    csv_path_three = "/Users/kevinhe/orioles-project/data/out/visitor_vs_home_first_inning.csv"
    csv_first_inning_decade = "/Users/kevinhe/orioles-project/first_inning_nb_summary_by_decade.csv"

    df = pd.read_csv(csv_path)
    df_2 = pd.read_csv(csv_path_two)
    df_3 = pd.read_csv(csv_path_three)

    # Filter for 2013–2024 if year column exists
    if "year" in df.columns:
        df = df[(df["year"] >= 2013) & (df["year"] <= 2024)]
    if "year" in df_2.columns:
        df_2 = df_2[(df_2["year"] >= 2013) & (df_2["year"] <= 2024)]

    # Group by inning
    df_grouped = df.groupby("inning")[["visitor_avg_runs", "home_avg_runs"]].mean().reset_index()

    # Generate plots
    plot_runs_per_inning(df_grouped)
    home_versus_visiting_inning(df_grouped)
    plot_differential_runs_per_inning(df_grouped)
    plot_differential_runs_first_inning(df_2)
    plot_home_vs_visitor_first_inning_line(df_3)
    plot_average_travel_bar()
    plot_first_inning_summary_by_decade(csv_first_inning_decade)

if __name__ == "__main__":
    main()