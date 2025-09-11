import pandas as pd
from pathlib import Path

def parse_csv_like(line: str):
    return [part.strip() for part in line.strip().split(",")]

POS_MAP = {
    "1": "P", "2": "C", "3": "1B", "4": "2B", "5": "3B",
    "6": "SS", "7": "LF", "8": "CF", "9": "RF", "10": "DH"
}

def parse_event_file(path: Path):
    games, roster, plays = [], [], []
    game_id, game_info = None, {}

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
                if game_id: flush_game()
                game_id = parts[1]
                game_info = {}

            elif rec == "info" and len(parts) >= 3:
                key, val = parts[1], ",".join(parts[2:])
                game_info[key] = val

            elif rec in ["start", "sub"] and len(parts) >= 5:
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
                inning, side, player_id, cnt, pitches, event = (
                    int(parts[1]), parts[2], parts[3], parts[4], parts[5], ",".join(parts[6:])
                )
                balls, strikes = None, None
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

    if game_id: flush_game()
    return games, roster, plays

def parse_event_folder(folder: str):
    games_all, roster_all, plays_all = [], [], []
    for p in Path(folder).glob("*.EV*"):
        g, r, pl = parse_event_file(p)
        games_all.extend(g)
        roster_all.extend(r)
        plays_all.extend(pl)
    return games_all, roster_all, plays_all

def main(folder, out):
    games, roster, plays = parse_event_folder(folder)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(games).to_csv(out_dir / "games.csv", index=False)
    pd.DataFrame(roster).to_csv(out_dir / "roster.csv", index=False)
    pd.DataFrame(plays).to_csv(out_dir / "plays.csv", index=False)
    print(f"[OK] Parsed {len(games)} games, {len(roster)} roster rows, {len(plays)} plays into {out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Folder with EVN/EVA files")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    main(args.folder, args.out)
