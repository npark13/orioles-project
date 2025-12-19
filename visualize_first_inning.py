import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def plot_runs_per_inning(df, save_path="all_visuals/retrosheet_remake_visuals/runs_per_inning.png"):
    df = df.dropna(subset=["inning", "visitor_avg_runs", "home_avg_runs", "games"])

    summary = []
    for inning in sorted(df["inning"].unique()):
        d = df[df["inning"] == inning]
        total_games = d["games"].sum()
        summary.append({
            "inning": inning,
            "visitor_avg": (d["visitor_avg_runs"] * d["games"]).sum() / total_games,
            "home_avg": (d["home_avg_runs"] * d["games"]).sum() / total_games,
        })

    df_plot = pd.DataFrame(summary)
    df_plot["total_avg_runs"] = df_plot["visitor_avg"] + df_plot["home_avg"]

    plt.figure(figsize=(10, 6))
    plt.plot(df_plot["inning"], df_plot["total_avg_runs"], marker="o", color="black")
    plt.xlabel("Inning")
    plt.ylabel("Average Runs per Inning")
    plt.title("Average Runs per Inning (Both Teams, 2013–2024)")
    plt.xticks(range(1, 11))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def home_versus_visiting_inning(df, save_path="all_visuals/retrosheet_remake_visuals/home_versus_visiting_inning.png"):
    df = df.dropna(subset=["inning", "visitor_avg_runs", "home_avg_runs", "games"])

    summary = []
    for inning in sorted(df["inning"].unique()):
        d = df[df["inning"] == inning]
        total_games = d["games"].sum()
        summary.append({
            "inning": inning,
            "visitor_avg": (d["visitor_avg_runs"] * d["games"]).sum() / total_games,
            "home_avg": (d["home_avg_runs"] * d["games"]).sum() / total_games,
        })

    df_plot = pd.DataFrame(summary)

    plt.figure(figsize=(10, 6))
    plt.plot(df_plot["inning"], df_plot["visitor_avg"], marker="o", label="Visitor")
    plt.plot(df_plot["inning"], df_plot["home_avg"], marker="o", label="Home")
    plt.xlabel("Inning")
    plt.ylabel("Runs per Inning")
    plt.title("Average Runs per Inning by Visitor and Home, 2013–2024")
    plt.xticks(range(1, 11))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_differential_runs_per_inning(df, save_path="all_visuals/retrosheet_remake_visuals/differential_runs_per_inning.png"):
    df = df.dropna(subset=["inning", "visitor_avg_runs", "home_avg_runs", "games"])

    summary = []
    for inning in sorted(df["inning"].unique()):
        d = df[df["inning"] == inning]
        total_games = d["games"].sum()
        diff = ((d["home_avg_runs"] - d["visitor_avg_runs"]) * d["games"]).sum() / total_games
        summary.append({"inning": inning, "diff_avg": diff})

    df_plot = pd.DataFrame(summary)

    plt.figure(figsize=(10, 6))
    plt.plot(df_plot["inning"], df_plot["diff_avg"], marker="o", color="black")
    plt.xlabel("Inning")
    plt.ylabel("Run Differential (Home − Visitor)")
    plt.title("Run Differential by Inning, 2013–2024")
    plt.xticks(range(1, 11))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_differential_runs_first_inning(df, save_path="all_visuals/retrosheet_remake_visuals/home_minus_vis_plot.png"):
    if not {"year", "home_minus_vis_runs"}.issubset(df.columns):
        return

    df = df.dropna(subset=["year", "home_minus_vis_runs"])
    has_games = "games" in df.columns

    def weighted_avg(d):
        w = d["games"] if has_games else 1
        return (d["home_minus_vis_runs"] * w).sum() / w.sum()

    summary = df.groupby("year").apply(weighted_avg).reset_index(name="diff")

    plt.figure(figsize=(10, 6))
    plt.plot(summary["year"], summary["diff"], marker="o", color="black")
    plt.xlabel("Year")
    plt.ylabel("Run Differential")
    plt.title("First Inning Run Differential by Year")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_first_inning_diff_by_decade(df, save_path="all_visuals/retrosheet_remake_visuals/first_inning_diff_by_decade.png"):
    df = df.copy()
    df["decade"] = (df["year"] // 10) * 10

    summary = []
    for dec, d in df.groupby("decade"):
        total_games = d["games"].sum()
        diff = ((d["home_avg_runs_1st"] - d["visitor_avg_runs_1st"]) * d["games"]).sum() / total_games
        summary.append({"decade": dec, "diff_avg": diff})

    df_plot = pd.DataFrame(summary)

    plt.figure(figsize=(10, 6))
    plt.plot(df_plot["decade"].astype(str) + "s", df_plot["diff_avg"], marker="o", color="black")
    plt.xlabel("Decade")
    plt.ylabel("Average First-Inning Run Differential")
    plt.title("First Inning Run Differential by Decade")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="all_visuals/retrosheet_remake_visuals")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("data/out/inning_summary.csv")
    df_2 = pd.read_csv("data/out/first_inning_summary.csv")

    if "year" in df.columns:
        df = df[(df["year"] >= 2013) & (df["year"] <= 2024)]

    plot_first_inning_diff_by_decade(df_2)
    plot_runs_per_inning(df)
    home_versus_visiting_inning(df)
    plot_differential_runs_per_inning(df)
    plot_differential_runs_first_inning(df_2)


if __name__ == "__main__":
    main()