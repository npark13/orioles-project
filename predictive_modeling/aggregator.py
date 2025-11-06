import pandas as pd
import glob

# Path to all yearly CSVs
files = glob.glob('/Users/kevinhe/orioles-project/data/out/rolling_avg/game_level_with_rolling_avg_weather_*.csv')

# Load and concatenate
df_list = []
for f in files:
    df_list.append(pd.read_csv(f))

df = pd.concat(df_list, ignore_index=True)

# Optional: save combined file
df.to_csv('/Users/kevinhe/orioles-project/data/out/rolling_avg/game_level_with_rolling_avg_weather_2014_2024.csv', index=False)
