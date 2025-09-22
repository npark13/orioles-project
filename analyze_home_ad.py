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

def summarize_year_all_innings(plays_csv: Path) -> dict:
    df = pd.read_csv(plays_csv)
    if "inning" not in df.columns or "batting_home" not in df.columns:
        return {}
    
    df["runs_scored"] = df["event_raw"].apply(runs_from_event)
    
    # Group by game, home/visitor, and inning
    g = df.groupby(["game_id", "batting_home", "inning"])["runs_scored"].sum().reset_index()
    
    summary_rows = []
    
    # Iterate over innings to compute statistics
    for inning in sorted(df["inning"].unique()):
        inning_data = g[g["inning"] == inning]
        vis = inning_data[inning_data["batting_home"] == 0].groupby("game_id")["runs_scored"].sum()
        home = inning_data[inning_data["batting_home"] == 1].groupby("game_id")["runs_scored"].sum()
        summary = pd.DataFrame({"visitor_runs": vis, "home_runs": home}).fillna(0)
        
        summary_rows.append({
            "inning": inning,
            "games": len(summary),
            "visitor_avg_runs": summary["visitor_runs"].mean(),
            "home_avg_runs": summary["home_runs"].mean(),
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
        stats_list = summarize_year_all_innings(plays_csv)
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