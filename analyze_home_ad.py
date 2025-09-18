
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

def summarize_year(plays_csv: Path) -> dict:
    df = pd.read_csv(plays_csv)
    if "inning" not in df.columns or "batting_home" not in df.columns:
        return {}
    first = df[df["inning"] == 1].copy()
    if first.empty:
        return {}
    first["runs_scored"] = first["event_raw"].apply(runs_from_event)
    g = (
        first.groupby(["game_id", "batting_home"])["runs_scored"]
        .sum()
        .reset_index()
    )
    vis = g[g["batting_home"] == 0].groupby("game_id")["runs_scored"].sum()
    home = g[g["batting_home"] == 1].groupby("game_id")["runs_scored"].sum()
    summary = pd.DataFrame({"visitor_runs": vis, "home_runs": home}).fillna(0)
    return {
        "games": len(summary),
        "visitor_avg_runs_1st": summary["visitor_runs"].mean(),
        "home_avg_runs_1st": summary["home_runs"].mean(),
        "visitor_p_scored_1st": (summary["visitor_runs"] > 0).mean(),
        "home_p_scored_1st": (summary["home_runs"] > 0).mean(),
        "home_minus_vis_runs": summary["home_runs"].mean() - summary["visitor_runs"].mean(),
        "home_minus_vis_prob": (summary["home_runs"] > 0).mean() - (summary["visitor_runs"] > 0).mean(),
    }

def summarize_all(out_root: Path) -> Path:
    rows = []
    for year_dir in sorted(out_root.glob("[12][0-9][0-9][0-9]")):
        plays_csv = year_dir / "plays.csv"
        if not plays_csv.exists():
            continue
        stats = summarize_year(plays_csv)
        if stats:
            stats["year"] = year_dir.name
            rows.append(stats)
    if not rows:
        print("[WARN] no year summaries written")
        return out_root / "first_inning_summary.csv"
    out_df = pd.DataFrame(rows).sort_values("year")
    out_path = out_root / "first_inning_summary.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path}")
    return out_path

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Analyze 1st-inning home advantage")
    ap.add_argument("--plays_csv", help="Path to single plays.csv")
    ap.add_argument("--all_years", action="store_true", help="Scan data/out/<year>/plays.csv")
    ap.add_argument("--out_root", default="data/out")
    args = ap.parse_args()

    if args.all_years:
        summarize_all(Path(args.out_root))
    else:
        if not args.plays_csv:
            raise SystemExit("Pass --plays_csv or use --all_years")
        stats = summarize_year(Path(args.plays_csv))
        if not stats:
            print("[WARN] no stats produced")
        else:
            for k, v in stats.items():
                print(f"{k}: {v}")
