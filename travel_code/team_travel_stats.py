#!/usr/bin/env python3
"""
team_travel_stats.py

Parses `data/team_travel.csv` which has blocks like:

2019
BAL, 24567
NYY, 28765
...

and writes `data/team_travel_stats_with_teams.csv` containing, per year:
Year, Low_Team, Low, Median_Team, Median, High_Team, High, Average
"""

from pathlib import Path
import csv
import statistics

# ---------- Project-aware paths (relative & portable) ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../orioles-project
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "team_travel.csv"
OUTPUT_FILE = DATA_DIR / "team_travel_stats_with_teams.csv"

def parse_year_blocks(lines):
    year_data = {}
    current_year = None
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.isdigit():
            current_year = int(s)
            year_data[current_year] = []
            continue
        try:
            team, miles = s.split(",")
            team = team.strip()
            miles = int(miles.strip())
            if current_year is not None:
                year_data[current_year].append((team, miles))
        except ValueError:
            # Skip malformed lines
            continue
    return year_data

def main():
    if not INPUT_FILE.exists():
        raise SystemExit(
            f"Missing input CSV: {INPUT_FILE}\n"
            f"Put team_travel.csv under {DATA_DIR}"
        )

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        lines = [line for line in f]

    year_data = parse_year_blocks(lines)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Year", "Low_Team", "Low", "Median_Team", "Median", "High_Team", "High", "Average"])

        for year in sorted(year_data.keys()):
            data = year_data[year]
            if not data:
                continue

            # Sort by miles
            sorted_data = sorted(data, key=lambda x: x[1])
            low_team, low = sorted_data[0]
            high_team, high = sorted_data[-1]

            miles_only = [m for _, m in sorted_data]
            median_value = int(statistics.median(miles_only))
            median_team = min(sorted_data, key=lambda x: abs(x[1] - median_value))[0]
            average = round(sum(miles_only) / len(miles_only), 2)

            writer.writerow([year, low_team, low, median_team, median_value, high_team, high, average])

    print(f"[OK] Statistics with team names saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
