import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_runs_per_inning(df, save_path="runs_per_inning.png"):
    # Drop missing/empty rows
    df = df.dropna(subset=["inning", "visitor_avg_runs", "home_avg_runs"])

    # Calculate combined runs (both teams)
    df["total_avg_runs"] = df["visitor_avg_runs"] + df["home_avg_runs"]

    innings = df["inning"]
    total_avg_runs = df["total_avg_runs"]

    plt.figure(figsize=(10, 6))
    plt.plot(innings, total_avg_runs, marker="o", linestyle="-", color="black")
    plt.xlabel("Inning")
    plt.ylabel("Average Runs per Inning")
    plt.title("Average Runs per Inning (Both Teams, 2013-2024)")
    plt.xticks(range(1, 11))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def home_versus_visiting_inning(df, save_path="home_versus_visiting_inning.png"):
    innings = df["inning"]
    visitor_avg_runs = df["visitor_avg_runs"]
    home_avg_runs = df["home_avg_runs"]

    plt.figure(figsize=(10, 6))
    plt.plot(innings, visitor_avg_runs, marker="o", linestyle="-", color="blue")
    plt.plot(innings, home_avg_runs, marker="s", linestyle="-", color="red")
    plt.xlabel("Inning")
    plt.ylabel("Runs per Inning")
    plt.title("Average Runs per Inning by Visitor and Home, 2013-2024")
    plt.xticks(range(1, 11))
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_differential_runs_per_inning(df, save_path="differential_runs_per_inning.png"):
    # Drop missing/empty rows
    df = df.dropna(subset=["inning", "visitor_avg_runs", "home_avg_runs"])

    # Calculate combined runs (both teams)
    df["differential_avg_runs"] = df["home_avg_runs"] - df["visitor_avg_runs"]  

    innings = df["inning"]
    differential_avg_runs = df["differential_avg_runs"]

    plt.figure(figsize=(10, 6))
    plt.plot(innings, differential_avg_runs, marker="o", linestyle="-", color="black")
    plt.xlabel("Inning")
    plt.ylabel("Run Differential")
    plt.title("Run Differential (Home minus Visitor) by Inning, 2013-2024")
    plt.xticks(range(1, 11))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_first_inning_run_diff_by_decade(df, save_path="first_inning_diff_by_decade.png"):
    # Filter for first inning only
    first_inning = df[df["inning"] == 1].copy()

    # Create decade column
    first_inning["decade"] = (first_inning["year"].astype(int) // 10) * 10

    # Group by decade and calculate mean run differential
    decade_stats = first_inning.groupby("decade")["home_minus_vis_runs"].mean().reset_index()

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(decade_stats["decade"].astype(str), decade_stats["home_minus_vis_runs"], marker="o", color="navy")
    plt.xlabel("Decade")
    plt.ylabel("Average First Inning Run Differential (Home - Visitor)")
    plt.title("First Inning Run Differential by Decade (Home minus Visitor)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_differential_runs_first_inning(df, save_path="differential_runs_first_inning.png"):
    # Drop missing/empty rows
    differential_innings = df["home_minus_vis_runs"]
    years = df["year"]

    plt.figure(figsize=(10, 6))
    plt.plot(years, differential_innings, marker="o", linestyle="-", color="black")
    plt.xlabel("Year")
    plt.ylabel("Run Differential")
    plt.title("First Inning run differential by year, 1909-2024")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_home_vs_visitor_first_inning_line(df, save_path: str = "home_vs_visitor_first_inning.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(df["visitor_bin"], df["avg_home_runs_first_inning"], marker='o', linestyle='-')
    plt.xlabel("Visitor Runs in First Inning")
    plt.ylabel("Average Home Team Runs in First Inning")
    plt.title("Correlation of Home Scoring and Visitor Scoring")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



def main():
    # Load inning summary CSV
    csv_path = "/Users/kevinhe/orioles-project/data/out/inning_summary.csv"
    csv_path_two = "/Users/kevinhe/orioles-project/data/out/first_inning_summary.csv"
    csv_path_three = "/Users/kevinhe/orioles-project/data/out/visitor_vs_home_first_inning.csv"

    df = pd.read_csv(csv_path)
    df_2 = pd.read_csv(csv_path_two)
    df_3 = pd.read_csv(csv_path_three)



    # restrict to 2010–present
    df = df[(df["year"] <= 2024) & (df["year"] >= 2013)]

    # group by inning and average across all years
    df_grouped = df.groupby("inning")[["visitor_avg_runs", "home_avg_runs"]].mean().reset_index()

    # single plot
    plot_runs_per_inning(df_grouped)
    home_versus_visiting_inning(df_grouped)
    plot_differential_runs_per_inning(df_grouped)
    plot_differential_runs_first_inning(df_2)
    plot_home_vs_visitor_first_inning_line(df_3)




if __name__ == "__main__":
    main()