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

def plot_average_travel(csv_path="/Users/kevinhe/orioles-project/average_travel_2020s.csv",
                        save_path="average_travel_2014_2024.png"):
    # Load data
    df = pd.read_csv(csv_path)

    # Filter for 2014–2024
    df_filtered = df[(df["year"] >= 2014) & (df["year"] <= 2024)]

    # Convert km to miles
    df_filtered["avg_travel_miles"] = df_filtered["avg_travel_km"] * 0.621371

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(df_filtered["year"].astype(str), df_filtered["avg_travel_miles"], color="skyblue")
    plt.xlabel("Year")
    plt.ylabel("Average Miles Traveled per Team")
    plt.title("Average Miles Traveled per Team (2014–2024)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def main():
    # Load CSVs
    csv_path = "/Users/kevinhe/orioles-project/data/out/inning_summary.csv"
    csv_path_two = "/Users/kevinhe/orioles-project/data/out/first_inning_summary.csv"
    csv_path_three = "/Users/kevinhe/orioles-project/data/out/visitor_vs_home_first_inning.csv"

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
    plot_average_travel()

if __name__ == "__main__":
    main()