from __future__ import annotations
from pathlib import Path
import pandas as pd

def add_names_for_year(year_dir: Path):
    rosters_path = year_dir / "rosters.csv"
    plays_path = year_dir / "plays.csv"
    roster_events_path = year_dir / "roster.csv"
    if not (rosters_path.exists() and plays_path.exists() and roster_events_path.exists()):
        return

    rosters = pd.read_csv(rosters_path)
    plays = pd.read_csv(plays_path)
    roster_events = pd.read_csv(roster_events_path)

    plays_named = plays.merge(
        rosters[["player_id", "player_name", "bats", "throws"]],
        how="left", left_on="batter_id", right_on="player_id"
    ).drop(columns=["player_id"])

    roster_named = roster_events.merge(
        rosters[["player_id", "player_name", "team_id", "pos"]],
        how="left", on="player_id"
    )

    plays_named.to_csv(year_dir / "plays_named.csv", index=False)
    roster_named.to_csv(year_dir / "roster_named.csv", index=False)
    print(f"[OK] {year_dir.name}: wrote plays_named.csv, roster_named.csv")

def main(out_root: str = "data/out"):
    out_root_p = Path(out_root)
    for year_dir in sorted(out_root_p.glob("[12][0-9][0-9][0-9]")):
        add_names_for_year(year_dir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Join names onto per-year outputs")
    ap.add_argument("--out_root", default="data/out")
    args = ap.parse_args()
    main(args.out_root)
