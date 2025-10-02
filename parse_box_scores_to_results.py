import csv
from pathlib import Path

# Repo root assumed to be location of this script
ROOT = Path(__file__).resolve().parent

# Folders that contain the box score text files
BOX_DIRS = [
    ROOT / "data" / "2010sbox",
    ROOT / "data" / "2020sbox",
]

OUT = ROOT / "results_by_game.csv"

START_YEAR, END_YEAR = 2013, 2024

def parse_box_file(path: Path, out_dict: dict):
    """
    Parse a Retrosheet-style box score flat file that may contain multiple games.
    We detect a new game when a line starts with 'id,'.
    We take final totals from 'line,0,...' (visitor) and 'line,1,...' (home),
    where the last field is the team’s final runs.
    """
    game = None  # current game accumulator

    def flush_game():
        if not game:
            return
        # require minimum fields
        if (
            game.get("game_id")
            and game.get("visteam")
            and game.get("hometeam")
            and game.get("date")
            and isinstance(game.get("visitor_final"), int)
            and isinstance(game.get("home_final"), int)
        ):
            # Only keep games in the requested year range
            # date format is typically 'YYYY/MM/DD'
            date = game["date"]
            y = None
            try:
                y = int(str(date).split("/")[0])
            except Exception:
                pass
            if y is not None and START_YEAR <= y <= END_YEAR:
                gid = game["game_id"]
                out_dict[gid] = {
                    "game_id": gid,
                    "visteam": game["visteam"],
                    "hometeam": game["hometeam"],
                    "date": date,
                    "visitor_final": game["visitor_final"],
                    "home_final": game["home_final"],
                }

    # Read file line-by-line with csv.reader to respect quoted commas
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            tag = row[0].strip().lower()

            if tag == "id":
                # New game starts; flush previous
                flush_game()
                game = {
                    "game_id": row[1].strip() if len(row) > 1 else None,
                    "visteam": None,
                    "hometeam": None,
                    "date": None,
                    "visitor_final": None,
                    "home_final": None,
                }
                continue

            if not game:
                # Skip until we see the first 'id'
                continue

            if tag == "info" and len(row) >= 3:
                key = row[1].strip().lower()
                val = (row[2] or "").strip()
                if key in ("visteam", "hometeam", "date"):
                    game[key] = val
                continue

            if tag == "line" and len(row) >= 3:
                # row format: line,teamIndex, runs_by_inning..., TOTAL
                # last column is the total runs
                try:
                    team_idx = int(row[1])
                    total = int(row[-1])
                except Exception:
                    continue
                if team_idx == 0:
                    game["visitor_final"] = total
                elif team_idx == 1:
                    game["home_final"] = total
                continue

        # flush last game in file
        flush_game()

def main():
    # Gather all box files from the decade dirs
    box_files = []
    for d in BOX_DIRS:
        if d.exists():
            # include everything that looks like a text/box score file
            # (Retrosheet box files often have no extension or .EBX)
            for p in d.rglob("*"):
                if p.is_file():
                    # crude filter: avoid obvious non-text (csv, png, etc.)
                    if p.suffix.lower() in (".csv", ".png", ".jpg", ".jpeg", ".pdf"):
                        continue
                    box_files.append(p)

    if not box_files:
        raise SystemExit("No box score files found under data/2010sbox or data/2020sbox.")

    results = {}
    for f in box_files:
        parse_box_file(f, results)

    if not results:
        raise SystemExit(f"No parsed games between {START_YEAR}-{END_YEAR} were found in box files.")

    # Write unified results_by_game.csv
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "visteam", "hometeam", "date", "visitor_final", "home_final"])
        for gid in sorted(results.keys()):
            r = results[gid]
            w.writerow([r["game_id"], r["visteam"], r["hometeam"], r["date"], r["visitor_final"], r["home_final"]])

    print(f"[OK] Wrote {OUT} with {len(results):,} games from {START_YEAR}-{END_YEAR}")

if __name__ == "__main__":
    main()
