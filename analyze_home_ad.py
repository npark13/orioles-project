from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

def runs_from_event(event: str) -> int:
    if not isinstance(event, str):
        return 0
    runs = len(re.findall(r"-H\b", event))
    if event.startswith("HR") and not re.search(r"\bB-?H\b", event):
        runs += 1
    return runs

def summarize_year(plays_csv: Path) -> list[dict]:
    # Read only necessary columns, keep event_raw as string
    df = pd.read_csv(
        plays_csv,
        usecols=["game_id", "inning", "batting_home", "event_raw"],
        dtype={"event_raw": str},
        low_memory=False
    )

    # Safely convert numeric columns
    df["inning"] = pd.to_numeric(df["inning"], errors="coerce").astype("Int64")
    df["batting_home"] = pd.to_numeric(df["batting_home"], errors="coerce").astype("Int64")

    # Drop rows with missing critical data
    df = df.dropna(subset=["inning", "batting_home", "event_raw"])

    # Only keep first 10 innings
    df = df[df["inning"] <= 10].copy()

    # Calculate runs scored in each event
    df["runs_scored"] = df["event_raw"].apply(runs_from_event)

    # Group by game, home/visitor, and inning
    g = df.groupby(["game_id", "batting_home", "inning"])["runs_scored"].sum().reset_index()

    summary_rows = []

    # Iterate explicitly over innings 1–10
    for inning in range(1, 11):
        inning_data = g[g["inning"] == inning]
        if inning_data.empty:
            continue

        # Make sure all games are included, even if a team didn't bat
        all_games = inning_data["game_id"].unique()
        vis = inning_data[inning_data["batting_home"] == 0].groupby("game_id")["runs_scored"].sum()
        home = inning_data[inning_data["batting_home"] == 1].groupby("game_id")["runs_scored"].sum()
        summary = pd.DataFrame({"visitor_runs": vis, "home_runs": home})


        summary_rows.append({
            "inning": inning,
            "games": len(summary),
            "visitor_avg_runs": summary["visitor_runs"].mean(skipna=True),
        "home_avg_runs": summary["home_runs"].mean(skipna=True),
            "visitor_p_scored": (summary["visitor_runs"] > 0).mean(),
            "home_p_scored": (summary["home_runs"] > 0).mean(),
            "home_minus_vis_runs": summary["home_runs"].mean() - summary["visitor_runs"].mean(),
            "home_minus_vis_prob": (summary["home_runs"] > 0).mean() - (summary["visitor_runs"] > 0).mean(),
        })

    return summary_rows

def summarize_all_innings(out_root: Path) -> Path:
    rows = []
    for year_dir in sorted(out_root.glob("[12][0-9][0-9][0-9]")):
        plays_csv = year_dir / "plays.csv"
        if not plays_csv.exists():
            continue
        stats_list = summarize_year(plays_csv)
        if stats_list:
            for stats in stats_list:
                stats["year"] = year_dir.name
                rows.append(stats)
    if not rows:
        print("[WARN] no inning summaries written")
        return out_root / "inning_summary.csv"
    
    out_df = pd.DataFrame(rows).sort_values(["year", "inning"])
    out_path = out_root / "inning_summary.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path}")
    return out_path

def summarize_first_inning_summary(out_root: Path) -> Path:
    """
    Writes per-year first-inning summary to:
      out_root / "first_inning_summary.csv"

    Exact columns (and order):
      games,visitor_avg_runs_1st,home_avg_runs_1st,visitor_p_scored_1st,home_p_scored_1st,
      home_minus_vis_runs,home_minus_vis_prob,year
    """
    rows = []

    for year_dir in sorted(out_root.glob("[12][0-9][0-9][0-9]")):
        plays_csv = year_dir / "plays.csv"
        if not plays_csv.exists():
            continue

        stats_list = summarize_year(plays_csv)
        if not stats_list:
            continue

        first = next((r for r in stats_list if r.get("inning") == 1), None)
        if not first:
            continue

        rows.append({
            "games": first["games"],
            "visitor_avg_runs_1st": first["visitor_avg_runs"],
            "home_avg_runs_1st": first["home_avg_runs"],
            "visitor_p_scored_1st": first["visitor_p_scored"],
            "home_p_scored_1st": first["home_p_scored"],
            "home_minus_vis_runs": first["home_minus_vis_runs"],
            "home_minus_vis_prob": first["home_minus_vis_prob"],
            "year": int(year_dir.name),
        })

    out_path = out_root / "first_inning_summary.csv"
    if not rows:
        print("[WARN] no first-inning summaries written")
        # still create an empty file with the exact header, if you want:
        pd.DataFrame(columns=[
            "games",
            "visitor_avg_runs_1st",
            "home_avg_runs_1st",
            "visitor_p_scored_1st",
            "home_p_scored_1st",
            "home_minus_vis_runs",
            "home_minus_vis_prob",
            "year",
        ]).to_csv(out_path, index=False)
        return out_path

    df_out = pd.DataFrame(rows)

    # enforce exact column order
    df_out = df_out[[
        "games",
        "visitor_avg_runs_1st",
        "home_avg_runs_1st",
        "visitor_p_scored_1st",
        "home_p_scored_1st",
        "home_minus_vis_runs",
        "home_minus_vis_prob",
        "year",
    ]].sort_values("year")

    df_out.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path}")
    return out_path