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
    plt.title("Average Runs per Inning (Both Teams, 1909–2024)")
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
    plt.title("Average Runs per Inning by Visitor and Home, 1909–2024")
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
    plt.title("Run Differential (Home minus Visitor) by Inning, 1909-2024")
    plt.xticks(range(1, 11))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def main():
    # Load inning summary CSV
    csv_path = "/Users/kevinhe/orioles-project/data/out/inning_summary.csv"
    df = pd.read_csv(csv_path)

    # restrict to 2010–present
    df = df[(df["year"] <= 2024) & (df["year"] >= 1909)]

    # group by inning and average across all years
    df_grouped = df.groupby("inning")[["visitor_avg_runs", "home_avg_runs"]].mean().reset_index()

    # single plot
    plot_runs_per_inning(df_grouped)
    home_versus_visiting_inning(df_grouped)
    plot_differential_runs_per_inning(df_grouped)

if __name__ == "__main__":
    main()