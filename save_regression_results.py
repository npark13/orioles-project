# save_regression_results.py
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

IN_CSV = "first_inning_features_with_tempo_all.csv"  # change if you used a different filename

# 1) Load
df = pd.read_csv(IN_CSV)

# 2) Target: did home team score in bottom 1st?
if "home_runs_bottom1" not in df.columns:
    raise SystemExit("Expected column 'home_runs_bottom1' is missing in your input CSV.")
df["home_scored_b1"] = (pd.to_numeric(df["home_runs_bottom1"], errors="coerce").fillna(0) > 0).astype(int)

# 3) Choose elapsed column
elapsed_col = (
    "top1_elapsed_seconds_final"
    if "top1_elapsed_seconds_final" in df.columns
    else ("top1_elapsed_seconds" if "top1_elapsed_seconds" in df.columns else None)
)
if not elapsed_col:
    raise SystemExit("Could not find an elapsed time column (top1_elapsed_seconds_final or top1_elapsed_seconds).")

# 4) Features (only keep those that exist)
candidate_X = ["visitor_batters_top1", "visitor_pitches_top1", elapsed_col]
use_cols = [c for c in candidate_X if c in df.columns]
if len(use_cols) < 2:
    raise SystemExit(f"Not enough predictors available. Found only: {use_cols}")

# Build X / y and drop rows with missing predictors
X = df[use_cols].apply(pd.to_numeric, errors="coerce")
y = df["home_scored_b1"].astype(int)
mask = X.notna().all(axis=1) & y.notna()
X = X.loc[mask].copy()
y = y.loc[mask].copy()
X = sm.add_constant(X)

# 5) Fit logit
model = sm.Logit(y, X).fit(disp=False)

# 6) Prepare outputs
coef_df = pd.DataFrame({
    "coef": model.params,
    "std_err": model.bse,
    "z": model.tvalues,
    "p_value": model.pvalues,
})
ci = model.conf_int()
coef_df["conf_low"] = ci[0]
coef_df["conf_high"] = ci[1]

metrics = {
    "n_obs": int(model.nobs),
    "log_likelihood": float(model.llf),
    "ll_null": float(model.llnull),
    "pseudo_r2_mcFadden": float(1 - model.llf / model.llnull) if model.llnull != 0 else np.nan,
    "elapsed_column_used": elapsed_col,
    "mean_home_scored_b1": float(y.mean()),
}

# 7) Save
OUT_COEF = "logit_regression_results_coefficients.csv"
OUT_METR = "logit_regression_results_metrics.csv"
OUT_SUMM = "logit_regression_results_summary.txt"

coef_df.to_csv(OUT_COEF, index=True)
pd.DataFrame([metrics]).to_csv(OUT_METR, index=False)
with open(OUT_SUMM, "w") as f:
    f.write(model.summary().as_text())

print("Wrote:")
print(f"  - {OUT_COEF}  (coefficients & CIs)")
print(f"  - {OUT_METR}  (model-level metrics)")
print(f"  - {OUT_SUMM}  (pretty text summary)")
print("\nColumns used as predictors:", use_cols)
