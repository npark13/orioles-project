#!/usr/bin/env python3
"""
Parse Retrosheet event files (EVN/EVA/EV?) into tidy CSVs.

- Single-season mode:
    python parse_events.py data/2024eve --out data/out/2024
- Recursive (all decades/years under a root):
    python parse_events.py data --out data/out --recursive

Writes per-year:
    data/out/<year>/games.csv
    data/out/<year>/roster.csv
    data/out/<year>/plays.csv
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import pandas as pd

_VALID_MIN, _VALID_MAX = 1871, 2026

POS_MAP = {
    "1": "P", "2": "C", "3": "1B", "4": "2B", "5": "3B",
    "6": "SS", "7": "LF", "8": "CF", "9": "RF", "10": "DH"
}

def _pick_year_from_stem(stem: str) -> Optional[str]:
    # prefer a year at end of stem (e.g., BOS1911 -> 1911)
    m = re.search(r"(\d{4})$", stem)
    if m:
        y = int(m.group(1))
        if _VALID_MIN <= y <= _VALID_MAX:
            return str(y)
    # otherwise consider all overlapping 4-digit windows, keep plausible, take MAX
    raw = re.findall(r"(?=(\d{4}))", stem)
    cands = [int(x) for x in raw if _VALID_MIN <= int(x) <= _VALID_MAX]
    return str(max(cands)) if cands else None

def year_from_path(p: Path) -> Optional[str]:
    y = _pick_year_from_stem(p.stem)
    if y:
        return y
    # fallback: search parent folder names, nearest first
    for part in reversed(p.parts):
        y = _pick_year_from_stem(part)
        if y:
            return y
    return None

def parse_csv_like(line: str) -> List[str]:
    return [part.strip() for part in line.strip().split(",")]

def parse_event_file(path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    games: List[Dict[str, Any]] = []
    roster: List[Dict[str, Any]] = []
    plays: List[Dict[str, Any]] = []

    game_id: Optional[str] = None
    game_info: Dict[str, Any] = {}

    def flush_game():
        if game_id:
            row = {"game_id": game_id}
            row.update(game_info)
            games.append(row)

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            if not raw.strip():
                continue
            parts = parse_csv_like(raw)
            rec = parts[0].lower()

            if rec == "id":
                if game_id:
                    flush_game()
                game_id = parts[1] if len(parts) > 1 else None
                game_info = {}

            elif rec == "info" and len(parts) >= 3:
                key, val = parts[1], ",".join(parts[2:])
                game_info[key] = val

            elif rec in ("start", "sub") and len(parts) >= 5:
                roster.append({
                    "game_id": game_id,
                    "player_id": parts[1],
                    "is_home": 1 if parts[2] in ("1", "H", "home") else 0,
                    "batting_order": int(parts[3]) if parts[3].isdigit() else None,
                    "field_pos_code": parts[4],
                    "field_pos": POS_MAP.get(parts[4], parts[4]),
                    "event": rec
                })

            elif rec == "play" and len(parts) >= 7:
                inning = int(parts[1])
                side = parts[2]
                player_id = parts[3]
                cnt = parts[4]
                pitches = parts[5]
                event = ",".join(parts[6:])
                balls = strikes = None
                if len(cnt) == 2 and cnt.isdigit():
                    balls, strikes = int(cnt[0]), int(cnt[1])
                plays.append({
                    "game_id": game_id,
                    "inning": inning,
                    "batting_home": 1 if side in ("1", "H", "home") else 0,
                    "batter_id": player_id,
                    "count_raw": cnt,
                    "balls": balls,
                    "strikes": strikes,
                    "pitches": pitches,
                    "event_raw": event
                })

            elif rec == "data" and len(parts) >= 2:
                subtype = parts[1]
                payload = ",".join(parts[2:]) if len(parts) > 2 else ""
                plays.append({
                    "game_id": game_id,
                    "inning": None,
                    "batting_home": None,
                    "batter_id": None,
                    "count_raw": None,
                    "balls": None,
                    "strikes": None,
                    "pitches": None,
                    "event_raw": f"DATA:{subtype}:{payload}"
                })
            else:
                # ignore comments/unknowns
                pass

    if game_id:
        flush_game()
    return games, roster, plays

def parse_event_folder(folder: str):
    games_all: List[Dict[str, Any]] = []
    roster_all: List[Dict[str, Any]] = []
    plays_all: List[Dict[str, Any]] = []
    for p in Path(folder).glob("*.EV*"):
        g, r, pl = parse_event_file(p)
        games_all.extend(g); roster_all.extend(r); plays_all.extend(pl)
    return games_all, roster_all, plays_all

def parse_events_recursive(root: str, out_root: str = "data/out"):
    root = Path(root)
    out_root = Path(out_root)
    files = list(root.rglob("*.EV*"))
    if not files:
        print(f"[WARN] No EV? files found under {root.resolve()}")
        return

    by_year: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for p in files:
        yr = year_from_path(p)
        if not yr:
            print(f"[SKIP] Could not infer year for {p}")
            continue
        g, r, pl = parse_event_file(p)
        bucket = by_year.setdefault(yr, {"games": [], "roster": [], "plays": []})
        bucket["games"].extend(g); bucket["roster"].extend(r); bucket["plays"].extend(pl)

    for yr, data in sorted(by_year.items(), key=lambda kv: kv[0]):
        out_dir = out_root / yr
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(data["games"]).to_csv(out_dir / "games.csv", index=False)
        pd.DataFrame(data["roster"]).to_csv(out_dir / "roster.csv", index=False)
        pd.DataFrame(data["plays"]).to_csv(out_dir / "plays.csv", index=False)
        print(f"[OK] {yr}: games={len(data['games'])}, roster={len(data['roster'])}, plays={len(data['plays'])} → {out_dir}")

def main(folder: str, out: str):
    games, roster, plays = parse_event_folder(folder)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(games).to_csv(out_dir / "games.csv", index=False)
    pd.DataFrame(roster).to_csv(out_dir / "roster.csv", index=False)
    pd.DataFrame(plays).to_csv(out_dir / "plays.csv", index=False)
    print(f"[OK] Parsed {len(games)} games, {len(roster)} roster rows, {len(plays)} plays into {out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Parse Retrosheet EVN/EVA event files")
    ap.add_argument("folder", help="Folder (single season) or root if --recursive")
    ap.add_argument("--out", default="data/out")
    ap.add_argument("--recursive", action="store_true")
    args = ap.parse_args()
    if args.recursive:
        parse_events_recursive(args.folder, args.out)
    else:
        main(args.folder, args.out)
