import pandas as pd
from pathlib import Path

def add_names(events_out="out", rosters_csv="out/rosters.csv"):
    out = Path(events_out)
    plays = pd.read_csv(out / "plays.csv")
    roster_events = pd.read_csv(out / "roster.csv")
    rosters = pd.read_csv(rosters_csv)

    plays_named = plays.merge(
        rosters[["player_id", "player_name", "bats", "throws"]],
        how="left", left_on="batter_id", right_on="player_id"
    ).drop(columns=["player_id"])

    roster_named = roster_events.merge(
        rosters[["player_id", "player_name", "team_id", "pos"]],
        how="left", on="player_id"
    )

    plays_named.to_csv(out / "plays_named.csv", index=False)
    roster_named.to_csv(out / "roster_named.csv", index=False)
    print("[OK] wrote plays_named.csv and roster_named.csv")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--events_out", default="out")
    ap.add_argument("--rosters_csv", default="out/rosters.csv")
    args = ap.parse_args()
    add_names(args.events_out, args.rosters_csv)
