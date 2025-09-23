import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_runs_per_inning(df, year=None, save_path="runs_per_inning.png"):
    # If a year is specified, filter for it
    if year is not None:
        df = df[df["year"] == year]

    # Drop missing/empty rows
    df = df.dropna(subset=["inning", "visitor_avg_runs", "home_avg_runs"])

    # Calculate combined runs (both teams)
    df["total_avg_runs"] = df["visitor_avg_runs"] + df["home_avg_runs"]

    innings = df["inning"]
    total_avg_runs = df["total_avg_runs"]

    plt.figure(figsize=(10, 6))
    plt.plot(innings, total_avg_runs, marker="o", linestyle="-", color="black")
    plt.xlabel("Inning")
    plt.ylabel("Runs per Inning")
    title = f"Average Runs Scored per Inning (Both Teams){' - ' + str(year) if year else ''}"
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def main():
    # Load inning summary CSV
    csv_path = "/Users/kevinhe/orioles-project/data/out/inning_summary.csv"
    df = pd.read_csv(csv_path)

    # Plot for a single year (example: 2010)
    plot_runs_per_inning(df, year=2010)

    # Or, if you wanted to average across all years:
    df_grouped = df.groupby("inning")[["visitor_avg_runs", "home_avg_runs"]].mean().reset_index()
    df_grouped["year"] = "1909–2013 (avg)"  # label for reference
    plot_runs_per_inning(df_grouped, year=None)

if __name__ == "__main__":
    main()