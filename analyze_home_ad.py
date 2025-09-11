import pandas as pd
import re

def runs_from_event(event: str) -> int:
    if pd.isna(event): return 0
    runs = 0
    runs += len(re.findall(r"-H", event))   # runner scores
    if event.startswith("HR") or event.startswith("H"):
        runs += 1                           # batter scores
    return runs

def analyze(plays_csv="out/plays.csv"):
    plays = pd.read_csv(plays_csv)
    first_inning = plays[plays["inning"] == 1].copy()
    first_inning["runs_scored"] = first_inning["event_raw"].apply(runs_from_event)

    game_runs = (
        first_inning.groupby(["game_id", "batting_home"])["runs_scored"]
        .sum()
        .reset_index()
    )

    visitor = game_runs[game_runs["batting_home"] == 0].groupby("game_id")["runs_scored"].sum()
    home = game_runs[game_runs["batting_home"] == 1].groupby("game_id")["runs_scored"].sum()

    summary = pd.DataFrame({"visitor_runs": visitor, "home_runs": home}).fillna(0)

    print("Visitor average runs in 1st:", summary["visitor_runs"].mean())
    print("Home average runs in 1st:", summary["home_runs"].mean())
    print("Visitor % scoring in 1st:", (summary["visitor_runs"] > 0).mean())
    print("Home % scoring in 1st:", (summary["home_runs"] > 0).mean())

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plays_csv", default="out/plays.csv")
    args = ap.parse_args()
    analyze(args.plays_csv)
