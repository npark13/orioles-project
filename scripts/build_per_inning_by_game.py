#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

ROOT = Path(".")
CSV_DIR = ROOT / "csv_files"
OUT = CSV_DIR / "per_inning_by_game.csv"

# Candidate column names we can map from various sources
GAME_ID_CANDS = ["game_id", "gameid", "gid", "retrosheet_game_id"]
HOME_R1_CANDS = ["home_r1","home_runs_1st","home_first_inning_runs","h1","home_1","home1","home_inn1"]
VIS_R1_CANDS  = ["visitor_r1","vis_r1","visitor_runs_1st","vis_first_inning_runs","v1",
                 "away_1","away1","vis_1","visitor_1","visit_inn1"]

def pick(cols, cands):
    cols = [c.lower() for c in cols]
    for c in cands:
        if c in cols:
            return cols.index(c)
    return None

def try_extract(fp: Path):
    try:
        df = pd.read_csv(fp, low_memory=False)
    except Exception:
        return None
    cols = [c.lower() for c in df.columns]
    gi = pick(cols, GAME_ID_CANDS)
    hr = pick(cols, HOME_R1_CANDS)
    vr = pick(cols, VIS_R1_CANDS)
    if gi is None or hr is None or vr is None:
        return None
    use = df.iloc[:, [gi, hr, vr]].copy()
    use.columns = ["game_id","home_r1","visitor_r1"]
    # coerce to numeric safely; non-numeric -> 0
    for c in ["home_r1","visitor_r1"]:
        use[c] = pd.to_numeric(use[c], errors="coerce").fillna(0).astype(int)
    # basic sanity
    use = use.dropna(subset=["game_id"]).drop_duplicates("game_id")
    return use

def main():
    if not CSV_DIR.exists():
        raise SystemExit(f"Missing folder {CSV_DIR}")

    pieces = []
    for fp in sorted(CSV_DIR.glob("*.csv")):
        got = try_extract(fp)
        if got is not None and not got.empty:
            print(f"[OK] harvested {len(got):,} rows from {fp.name}")
            pieces.append(got)

    if not pieces:
        raise SystemExit(
            "Could not find any CSV in csv_files/ with game_id and 1st-inning columns.\n"
            "Look for a file that has something like home_r1/visitor_r1 or h1/v1 or home_1/away_1."
        )

    out = pd.concat(pieces, ignore_index=True).drop_duplicates("game_id")
    out["home_scores_1st"] = (out["home_r1"] > 0).astype(int)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"[DONE] wrote {OUT}  rows={len(out):,}")

if __name__ == "__main__":
    main()
