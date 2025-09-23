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
    plt.title("Average Runs per Inning (Both Teams, 2010–Present)")
    plt.xticks(range(1, 11))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def main():
    # Load inning summary CSV
    csv_path = "/Users/kevinhe/orioles-project/data/out/inning_summary.csv"
    df = pd.read_csv(csv_path)

    # restrict to 1909-2013
    df = df[(df["year"] >= 1909) & (df["year"] <= 2013)]

    # group by inning and average across all years
    df_grouped = df.groupby("inning")[["visitor_avg_runs", "home_avg_runs"]].mean().reset_index()

    # single plot
    plot_runs_per_inning(df_grouped)

if __name__ == "__main__":
    main()