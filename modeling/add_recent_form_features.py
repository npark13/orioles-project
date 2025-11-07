#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

root = Path("data/out")
frames = []

# Loop over all yearly folders 2014–2024
for year_dir in sorted(root.iterdir()):
    if not year_dir.is_dir():
        continue
    games_fp = year_dir / "games.csv"
    if not games_fp.exists():
        continue

    print(f"Processing {year_dir.name}...")
    games = pd.read_csv(games_fp)
    per_inning = pd.read_csv(root / "per_inning_by_game.csv")

    games.columns = [c.lower() for c in games.columns]
    per_inning.columns = [c.lower() for c in per_inning.columns]

    df = games.merge(per_inning[["game_id", "home_r1"]], on="game_id", how="left")
    df = df.sort_values(["hometeam", "date"])
    df["home_recent_form"] = (
        df.groupby("hometeam")["home_r1"]
          .transform(lambda s: s.rolling(window=5, min_periods=1).mean())
    )
    frames.append(df[["game_id", "hometeam", "home_recent_form"]])

# Combine all seasons
out = pd.concat(frames, ignore_index=True)
out.to_csv("data/out/home_recent_form.csv", index=False)
print("saved data/out/home_recent_form.csv (all seasons)")
