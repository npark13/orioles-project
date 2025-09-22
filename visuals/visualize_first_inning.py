import pandas as pd
import matplotlib.pyplot as plt
import os

# Function to plot visitor vs home average runs
def plot_avg_runs(df, save_path="avg_runs_plot.png"):
    innings = df["inning"]
    visitor_avg_runs = df["visitor_avg_runs"]
    home_avg_runs = df["home_avg_runs"]

    plt.figure(figsize=(10, 6))
    plt.plot(innings, visitor_avg_runs, label="Visitor Avg Runs", marker="o", linestyle="-")
    plt.plot(innings, home_avg_runs, label="Home Avg Runs", marker="s", linestyle="-")
    plt.xlabel("Year")
    plt.ylabel("Average Runs")
    plt.title("Average Runs per Inning 2010-2024")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# Function to plot home_minus_vis_runs
def plot_home_minus_vis(df, save_path="home_minus_vis_plot.png"):
    years = df["year"]
    home_minus_vis = df["home_minus_vis_runs"]

    plt.figure(figsize=(10, 6))
    plt.plot(years, home_minus_vis, label="Home Minus Visitor Runs", marker="o", linestyle="-", color="purple")
    plt.xlabel("Year")
    plt.ylabel("Home - Visitor Runs in 1st Inning")
    plt.title("Difference Between Home and Visitor Runs in 1st Inning by Year")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def main():
    # Get the CSV path relative to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = "/Users/kevinhe/orioles-project/data/out/first_inning_summary.csv"

    # Load CSV
    df = pd.read_csv(csv_path)

    # Plot the graphs
    plot_avg_runs(df)
    plot_home_minus_vis(df)

if __name__ == "__main__":
    main()