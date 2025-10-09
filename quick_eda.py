# quick_eda.py
import pandas as pd
import numpy as np

df = pd.read_csv("first_inning_features_with_tempo.csv").dropna(subset=["top1_elapsed_seconds"])
df["home_scored_b1"] = (df["home_runs_bottom1"] > 0).astype(int)

# quintiles of “bench time”
df["q"] = pd.qcut(df["top1_elapsed_seconds"], 5, labels=[1,2,3,4,5])

print("Mean home runs in B1 by quintile:")
print(df.groupby("q")["home_runs_bottom1"].mean().round(3))

print("\nHome scored (rate) by quintile:")
print(df.groupby("q")["home_scored_b1"].mean().round(3))

# correlation (rough)
print("\nCorr(top1_elapsed_seconds, home_scored_b1):",
      np.corrcoef(df["top1_elapsed_seconds"], df["home_scored_b1"])[0,1].round(3))
