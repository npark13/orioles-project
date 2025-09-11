import pandas as pd
from pathlib import Path

def parse_team_file(team_file: str) -> pd.DataFrame:
    rows = []
    with open(team_file, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) >= 2:
                rows.append({"team_id": parts[0], "team_name": ",".join(parts[1:])})
    return pd.DataFrame(rows)

def parse_ros_folder(folder: str) -> pd.DataFrame:
    rows = []
    for p in Path(folder).glob("*.ROS"):
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                ln = ln.strip()
                if not ln: continue
                parts = [x.strip() for x in ln.split(",")]
                if len(parts) < 7: continue
                rows.append({
                    "player_id": parts[0],
                    "last": parts[1],
                    "first": parts[2],
                    "bats": parts[3],
                    "throws": parts[4],
                    "team_id": parts[5],
                    "pos": parts[6],
                    "player_name": f"{parts[2]} {parts[1]}".strip()
                })
    return pd.DataFrame(rows)

def main(folder, teamfile, out):
    ros_df = parse_ros_folder(folder)
    team_df = parse_team_file(teamfile)
    ros_df = ros_df.merge(team_df, how="left", on="team_id")
    ros_df.to_csv(out, index=False)
    print(f"[OK] Parsed {len(ros_df)} roster rows → {out}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--teamfile", required=True)
    ap.add_argument("--out", default="out/rosters.csv")
    args = ap.parse_args()
    main(args.folder, args.teamfile, args.out)
