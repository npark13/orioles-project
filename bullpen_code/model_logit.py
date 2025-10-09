# model_logit.py
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("first_inning_features_with_tempo.csv")
df = df.dropna(subset=["top1_elapsed_seconds"])
df["home_scored_b1"] = (df["home_runs_bottom1"] > 0).astype(int)

X = df[["top1_elapsed_seconds", "visiting_starter_tempo_be"]].fillna(0.0)
X = sm.add_constant(X)
y = df["home_scored_b1"]

m = sm.Logit(y, X).fit(disp=False)
print(m.summary())
