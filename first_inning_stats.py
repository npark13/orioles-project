import pandas as pd
from pathlib import Path

# Base folder containing year folders
base_folder = Path("/Users/nancypark/sports-analytics-project/orioles-project/data/out")

# Initialize counters
n1 = 0  # total games
n2 = 0  # both teams 0 after 1st
n3 = 0  # visitor led
n4 = 0  # home led
n5 = 0  # tied not 0-0

def count_runs(events):
    """
    Count runs from event_raw strings.
    Every occurrence of '-H' or '.H' in the string counts as 1 run.
    """
    runs = 0
    for e in events.dropna():
        runs += e.count("-H")  # runner scoring from any base
        runs += e.count(".H")  # backup, just in case
    return runs


# Loop through years
for year in range(1911, 2025):
    year_folder = base_folder / str(year)
    plays_file = year_folder / "plays.csv"
    
    if not plays_file.exists():
        continue  # skip missing files

    # Load CSV, force event_raw to string to avoid mixed-type warnings
    df = pd.read_csv(plays_file, dtype={"event_raw": str}, low_memory=False)

    # Filter first inning
    df_first = df[df["inning"] == 1.0]

    if df_first.empty:
        continue

    # Sum runs per game by home/visitor
    home_runs = df_first[df_first["batting_home"] == 1.0].groupby("game_id")["event_raw"].apply(count_runs)
    visitor_runs = df_first[df_first["batting_home"] == 0.0].groupby("game_id")["event_raw"].apply(count_runs)

    # Merge home and visitor runs
    scores = pd.DataFrame({"home": home_runs, "visitor": visitor_runs}).fillna(0)

    # Update counters
    n1 += len(scores)
    n2 += ((scores["home"] == 0) & (scores["visitor"] == 0)).sum()
    n3 += (scores["visitor"] > scores["home"]).sum()
    n4 += (scores["home"] > scores["visitor"]).sum()
    n5 += ((scores["home"] == scores["visitor"]) & (scores["home"] != 0)).sum()


output_path = "/Users/nancypark/sports-analytics-project/orioles-project/out/first_inning_summary_stats.csv"

with open(output_path, "w") as f:
    f.write("Frequency of Scoring Differences after 1st Inning, 1911-2024\n")
    f.write(f"Games: {n1}\n")
    f.write(f"Times both teams scored 0: {n2}\n")
    f.write(f"Visitor led: {n3}\n")
    f.write(f"Home led: {n4}\n")
    f.write(f"Tied, not at 0-0: {n5}\n")

# # Print final table
# print("Frequency of Scoring Differences after 1st Inning, 1911-2024")
# print("-----------------------------------------------------------")
# print(f"Games: {n1}")
# print(f"Times both teams scored 0: {n2}")
# print(f"Visitor led: {n3}")
# print(f"Home led: {n4}")
# print(f"Tied, not at 0-0: {n5}")