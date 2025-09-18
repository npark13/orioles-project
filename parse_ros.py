from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd

_VALID_MIN, _VALID_MAX = 1871, 2026

def year_from_name(name: str) -> Optional[str]:
    stem = Path(name).stem
    m = re.search(r"(\d{4})$", stem)
    if m:
        y = int(m.group(1))
        if _VALID_MIN <= y <= _VALID_MAX:
            return str(y)
    raw = re.findall(r"(?=(\d{4}))", stem)
    cands = [int(x) for x in raw if _VALID_MIN <= int(x) <= _VALID_MAX]
    return str(max(cands)) if cands else None

def parse_team_file(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) >= 2:
                rows.append({"team_id": parts[0], "team_name": ",".join(parts[1:])})
    return pd.DataFrame(rows)

def parse_ros_file(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = [x.strip() for x in ln.split(",")]
            if len(parts) < 7:
                continue
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

def parse_rosters_recursive(root: str, out_root: str = "data/out"):
    root_p = Path(root)
    out_root_p = Path(out_root)

    ros_files = list(root_p.rglob("*.ROS"))
    team_files = list(root_p.rglob("TEAM*"))

    team_by_year: Dict[str, pd.DataFrame] = {}
    for t in team_files:
        yr = year_from_name(t.name) or year_from_name(t.parent.name)
        if yr:
            team_by_year[yr] = parse_team_file(t)

    ros_by_year: Dict[str, List[pd.DataFrame]] = {}
    for rf in ros_files:
        yr = year_from_name(rf.name) or year_from_name(rf.parent.name)
        if not yr:
            print(f"[SKIP] Could not infer year for {rf}")
            continue
        df = parse_ros_file(rf)
        if df.empty:
            continue
        df["year"] = yr
        ros_by_year.setdefault(yr, []).append(df)

    for yr, parts in sorted(ros_by_year.items(), key=lambda kv: kv[0]):
        out_dir = out_root_p / yr
        out_dir.mkdir(parents=True, exist_ok=True)
        ros = pd.concat(parts, ignore_index=True)
        if yr in team_by_year:
            ros = ros.merge(team_by_year[yr], how="left", on="team_id")
        ros.to_csv(out_dir / "rosters.csv", index=False)
        print(f"[OK] {yr}: rosters={len(ros)} → {out_dir/'rosters.csv'}")

def main(folder: str, teamfile: str, out: str):
    folder_p = Path(folder)
    ros = pd.DataFrame()
    for p in folder_p.glob("*.ROS"):
        ros = pd.concat([ros, parse_ros_file(p)], ignore_index=True)
    team_df = parse_team_file(Path(teamfile))
    ros = ros.merge(team_df, how="left", on="team_id")
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    ros.to_csv(out_p, index=False)
    print(f"[OK] Parsed {len(ros)} roster rows → {out_p}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Parse Retrosheet ROS/TEAM files")
    ap.add_argument("root", help="Folder (single-season) or root if --recursive")
    ap.add_argument("--teamfile", default=None, help="TEAMYYYY (single-season only)")
    ap.add_argument("--out", default="data/out/rosters.csv")
    ap.add_argument("--recursive", action="store_true")
    args = ap.parse_args()

    if args.recursive:
        parse_rosters_recursive(args.root, "data/out")
    else:
        if not args.teamfile:
            raise SystemExit("In non-recursive mode you must pass --teamfile")
        main(args.root, args.teamfile, args.out)
