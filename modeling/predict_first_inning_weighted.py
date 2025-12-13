#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weighted Logistic + Boosted Model for 1st-inning scoring.

Option A: train ONE model per run, controlled by --target:
  --target home  -> P(home team scores >=1 in 1st)
  --target away  -> P(away/visiting team scores >=1 in 1st)
  --target yrfi  -> P(any run scored in 1st inning by either team)

Uses a fixed classification threshold THR=0.50 for pred_{logit,boost}.

Outputs (to --outdir, default "."):
- modeling_dataset_weighted.csv
- model_metrics_weighted.txt
- roc_logit.png, roc_boost.png
- calib_logit.png, calib_boost.png
- feature_importances_boost.csv
- predictions_logit.csv, predictions_boost.csv
- logit_pipeline.joblib, boost_pipeline.joblib
"""

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    f1_score,
    average_precision_score,
    classification_report,
    RocCurveDisplay,
    confusion_matrix,
)
from sklearn.calibration import CalibrationDisplay

# Try to import boosted learners (optional)
XGB_OK = LGB_OK = False
try:
    from xgboost import XGBClassifier  # pip install xgboost
    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    from lightgbm import LGBMClassifier  # pip install lightgbm
    LGB_OK = True
except Exception:
    LGB_OK = False


# --------------------- helpers ---------------------

TEAM_COORDS = {
    "ARI": (33.4484, -112.0740), "ATL": (33.7490, -84.3880),  "BAL": (39.2904, -76.6122),
    "BOS": (42.3601, -71.0589),  "CHN": (41.8781, -87.6298),  "CHA": (41.8781, -87.6298),
    "CHC": (41.8781, -87.6298),  "CHW": (41.8781, -87.6298),  "CIN": (39.1031, -84.5120),
    "CLE": (41.4993, -81.6944),  "COL": (39.7392, -104.9903), "DET": (42.3314, -83.0458),
    "FLA": (25.7617, -80.1918),  "MIA": (25.7617, -80.1918),  "HOU": (29.7604, -95.3698),
    "KCA": (39.0997, -94.5786),  "KCR": (39.0997, -94.5786),  "ANA": (33.8366, -117.9143),
    "LAA": (33.8366, -117.9143), "LAN": (34.0522, -118.2437), "LAD": (34.0522, -118.2437),
    "MIL": (43.0389, -87.9065),  "MIN": (44.9778, -93.2650),  "MON": (45.5089, -73.5542),
    "NYN": (40.7128, -74.0060),  "NYM": (40.7128, -74.0060),  "NYA": (40.7128, -74.0060),
    "NYY": (40.7128, -74.0060),  "OAK": (37.8044, -122.2711), "PHI": (39.9526, -75.1652),
    "PIT": (40.4406, -79.9959),  "SDN": (32.7157, -117.1611), "SDP": (32.7157, -117.1611),
    "SEA": (47.6062, -122.3321), "SFN": (37.7749, -122.4194), "SFG": (37.7749, -122.4194),
    "SLN": (38.6270, -90.1994),  "STL": (38.6270, -90.1994),  "TBA": (27.9506, -82.4572),
    "TBR": (27.9506, -82.4572),  "TEX": (32.7555, -97.3308),  "TOR": (43.6532, -79.3832),
    "WAS": (38.9072, -77.0369),  "WSN": (38.9072, -77.0369),
}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2.0)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def tz_from_lon(lon: float) -> int:
    # crude US time zone bins by longitude
    if lon <= -125: return -8
    if lon <= -115: return -7
    if lon <= -100: return -6
    if lon <= -85:  return -5
    return -4

def load_games_range(games_root: Path, start: int, end: int) -> pd.DataFrame:
    frames = []
    for y in range(start, end + 1):
        fp = games_root / str(y) / "games.csv"
        if not fp.exists():
            continue
        g = pd.read_csv(fp, low_memory=False)
        g.columns = [c.lower() for c in g.columns]
        if "date" in g.columns:
            g["date"] = pd.to_datetime(g["date"], errors="coerce")
        if "hometeam" not in g.columns and "home" in g.columns:
            g["hometeam"] = g["home"]
        if "visteam" not in g.columns and "away" in g.columns:
            g["visteam"] = g["away"]
        if "site" not in g.columns and "park" in g.columns:
            g["site"] = g["park"]
        if "number" not in g.columns:
            g["number"] = 1

        keep = [c for c in ["game_id", "date", "hometeam", "visteam", "site", "number"] if c in g.columns]
        frames.append(g[keep])

    if not frames:
        raise SystemExit(f"No games.csv found under {games_root} for {start}-{end}")

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["game_id", "date", "hometeam", "visteam"])
    return out

def attach_travel_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["visteam", "date", "number"]).copy()
    last_loc, last_time = {}, {}

    travel_km, tz_dir, rest_hours = [], [], []

    for _, r in df.iterrows():
        home, vis = r["hometeam"], r["visteam"]

        if home not in TEAM_COORDS or vis not in TEAM_COORDS:
            travel_km.append(np.nan)
            tz_dir.append("unknown")
            rest_hours.append(np.nan)
            continue

        dest_lat, dest_lon = TEAM_COORDS[home]

        if vis in last_loc:
            prev_lat, prev_lon = last_loc[vis]
            dkm = haversine_km(prev_lat, prev_lon, dest_lat, dest_lon)
            tz_prev = tz_from_lon(prev_lon)
        else:
            dkm = np.nan
            tz_prev = tz_from_lon(dest_lon)

        tz_now = tz_from_lon(dest_lon)
        if tz_now > tz_prev:
            dirlab = "eastbound"
        elif tz_now < tz_prev:
            dirlab = "westbound"
        else:
            dirlab = "same_zone"

        if vis in last_time:
            rh = (r["date"] - last_time[vis]).total_seconds() / 3600.0
        else:
            rh = np.nan

        travel_km.append(dkm)
        tz_dir.append(dirlab)
        rest_hours.append(rh)

        last_loc[vis] = (dest_lat, dest_lon)
        last_time[vis] = r["date"]

    df["visitor_travel_km"] = pd.Series(travel_km, index=df.index)
    df["tz_dir"] = pd.Series(tz_dir, index=df.index)
    df["rest_hours_since_last"] = pd.Series(rest_hours, index=df.index)
    return df

def year_weights(dates: pd.Series, base: float = 0.90) -> np.ndarray:
    years = pd.to_datetime(dates).dt.year
    ymax = years.max()
    return np.power(base, ymax - years).astype(float)

def _normalize_run_cols(per_inning: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize schema to always provide:
      - home_r1
      - away_r1  (if possible)
    Accepts common alternatives.
    """
    per = per_inning.copy()
    per.columns = [c.lower() for c in per.columns]

    # home runs
    if "home_r1" not in per.columns:
        if "home_first_inning_runs" in per.columns:
            per = per.rename(columns={"home_first_inning_runs": "home_r1"})
        elif "home_r" in per.columns:
            per = per.rename(columns={"home_r": "home_r1"})

    # away/visiting runs
    if "away_r1" not in per.columns:
        if "visiting_first_inning_runs" in per.columns:
            per = per.rename(columns={"visiting_first_inning_runs": "away_r1"})
        elif "visitor_first_inning_runs" in per.columns:
            per = per.rename(columns={"visitor_first_inning_runs": "away_r1"})
        elif "vis_r1" in per.columns:
            per = per.rename(columns={"vis_r1": "away_r1"})
        elif "away_r" in per.columns:
            per = per.rename(columns={"away_r": "away_r1"})

    return per


# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2013)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--games-root", type=str, default="data/out")
    ap.add_argument("--per-inning", type=str, required=True,
                    help="CSV with per-inning runs; requires game_id and a home+away first-inning runs columns.")
    ap.add_argument("--outdir", type=str, default=".")
    ap.add_argument("--weight-base", type=float, default=0.90,
                    help="Year decay base; lower = more recent emphasis.")
    ap.add_argument("--target", choices=["home", "away", "yrfi"], default="home",
                    help="Which event to predict: home scores, away scores, or any run in 1st (YRFI).")
    args = ap.parse_args()

    THR = 0.50  # fixed classification threshold

    root = Path(".").resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load games and attach travel features
    games = load_games_range(root / args.games_root, args.start, args.end)
    games = attach_travel_features(games)

    # 2) Load per-inning + normalize schema
    per_inning = pd.read_csv(root / args.per_inning, engine="python", on_bad_lines="warn")
    per_inning = _normalize_run_cols(per_inning)

    if "game_id" not in per_inning.columns:
        raise SystemExit(
            "per-inning file is missing 'game_id'.\n"
            f"Columns present: {per_inning.columns.tolist()}"
        )
    if "home_r1" not in per_inning.columns:
        raise SystemExit(
            "per-inning file must include home first-inning runs.\n"
            "Expected 'home_r1' or 'home_first_inning_runs'.\n"
            f"Columns present: {per_inning.columns.tolist()}"
        )
    if args.target in ("away", "yrfi") and "away_r1" not in per_inning.columns:
        raise SystemExit(
            f"--target {args.target} requires away first-inning runs.\n"
            "Expected 'away_r1' or 'visiting_first_inning_runs' (or similar).\n"
            f"Columns present: {per_inning.columns.tolist()}"
        )

    # Optional: align pitcher naming if present (kept from your earlier renames)
    if "home_era" in per_inning.columns and "home_starter_era" not in per_inning.columns:
        per_inning = per_inning.rename(columns={"home_era": "home_starter_era"})

    # Extra columns that may exist and can be used as features
    extra_cols = [
        "vis_era",
        "home_travel",
        "vis_travel",
        "home_avg_prev",
        "away_avg_prev",
        "home_obp",
        "away_obp",
    ]
    present_extra = [c for c in extra_cols if c in per_inning.columns]

    # 3) Merge onto games
    merge_cols = ["game_id", "home_r1"] + (["away_r1"] if "away_r1" in per_inning.columns else []) + present_extra
    df = games.merge(per_inning[merge_cols], on="game_id", how="left")

    # Label y according to target
    if args.target == "home":
        df["y"] = (df["home_r1"].fillna(0) > 0).astype(int)
    elif args.target == "away":
        df["y"] = (df["away_r1"].fillna(0) > 0).astype(int)
    else:  # yrfi
        df["y"] = ((df["home_r1"].fillna(0) + df["away_r1"].fillna(0)) > 0).astype(int)

    # Force extra columns numeric
    for c in present_extra:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Optional: merge recent form (home-only feature if file exists)
    recent_fp = root / "data/out/home_recent_form.csv"
    if recent_fp.exists():
        recent = pd.read_csv(recent_fp)
        recent.columns = [c.lower() for c in recent.columns]
        if {"game_id", "home_recent_form"}.issubset(recent.columns):
            df = df.merge(recent[["game_id", "home_recent_form"]], on="game_id", how="left")
            print("✓ Added recent form feature")

    # 4) Clean & feature table
    # Keep rows with known label inputs
    df = df.dropna(subset=["home_r1"])
    if args.target in ("away", "yrfi"):
        df = df.dropna(subset=["away_r1"])

    df["visitor_travel_km"] = df["visitor_travel_km"].fillna(0).clip(0, 6000)
    df["rest_hours_since_last"] = df["rest_hours_since_last"].fillna(48).clip(0, 168)
    df["season"] = pd.to_datetime(df["date"], errors="coerce").dt.year

    features_num = ["visitor_travel_km", "rest_hours_since_last"]

    # include recent form if present
    if "home_recent_form" in df.columns:
        features_num.append("home_recent_form")

    # include any extra rolling/context vars that exist
    for extra in ["vis_era", "home_travel", "vis_travel", "home_avg_prev", "away_avg_prev", "home_obp", "away_obp"]:
        if extra in df.columns and extra not in features_num:
            features_num.append(extra)

    # categorical features
    # For away/yrfi it often helps to include visteam too; harmless for home as well.
    features_cat = ["tz_dir", "hometeam", "visteam"]

    keep_cols = list(dict.fromkeys(
        ["game_id", "date", "hometeam", "visteam", "y"] + features_num + features_cat
    ))
    model_df = df[keep_cols].copy()
    model_df.to_csv(outdir / "modeling_dataset_weighted.csv", index=False)

    # 5) Time-aware split (last ~1 season as test)
    yrs = pd.to_datetime(model_df["date"], errors="coerce").dt.year.dropna()
    if len(yrs.unique()) >= 3:
        cutoff = yrs.max() - 1
        train_idx = pd.to_datetime(model_df["date"], errors="coerce").dt.year <= cutoff
        test_idx  = pd.to_datetime(model_df["date"], errors="coerce").dt.year >  cutoff
    else:
        train_idx = pd.Series([True] * len(model_df), index=model_df.index)
        test_idx  = ~train_idx

    X = model_df[features_num + features_cat]
    y = model_df["y"].astype(int).values

    if test_idx.sum() == 0:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )
        w_tr = year_weights(model_df.loc[X_tr.index, "date"], base=args.weight_base)
        w_te = np.ones_like(y_te, dtype=float)
    else:
        X_tr, X_te = X.loc[train_idx], X.loc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        w_tr = year_weights(model_df.loc[train_idx, "date"], base=args.weight_base)
        w_te = np.ones_like(y_te, dtype=float)

    # 6) Preprocessing (impute → encode/scale)
    num_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_tf, features_num),
            ("cat", cat_tf, features_cat)
        ],
        remainder="drop",
        sparse_threshold=0
    )

    # 7) Weighted logistic regression
    logit = Pipeline(steps=[
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"))
    ])
    logit.fit(X_tr, y_tr, clf__sample_weight=w_tr)
    proba_l = logit.predict_proba(X_te)[:, 1]
    pred_l = (proba_l >= THR).astype(np.uint8)

    auc_l = roc_auc_score(y_te, proba_l)
    ll_l  = log_loss(y_te, proba_l)
    f1_l  = f1_score(y_te, pred_l)
    pr_l  = average_precision_score(y_te, proba_l)

    print(f"[DEBUG] LOGIT  target={args.target}  thr={THR:.2f}  predicted_positive_rate={pred_l.mean():.3f}")

    # 8) Boosted model (xgboost > lightgbm > random forest fallback)
    boost_name = "xgb" if XGB_OK else ("lgbm" if LGB_OK else "rf")

    if XGB_OK:
        boost = Pipeline(steps=[
            ("pre", pre),
            ("clf", XGBClassifier(
                n_estimators=600, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                objective="binary:logistic", eval_metric="logloss",
                random_state=42, n_jobs=-1
            ))
        ])
        boost.fit(X_tr, y_tr, clf__sample_weight=w_tr)
    elif LGB_OK:
        boost = Pipeline(steps=[
            ("pre", pre),
            ("clf", LGBMClassifier(
                n_estimators=800, max_depth=-1, num_leaves=31,
                learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, random_state=42, n_jobs=-1
            ))
        ])
        boost.fit(X_tr, y_tr, clf__sample_weight=w_tr)
    else:
        warnings.warn("XGBoost/LightGBM not available; falling back to RandomForest.")
        boost = Pipeline(steps=[
            ("pre", pre),
            ("clf", RandomForestClassifier(
                n_estimators=500, max_depth=None, min_samples_leaf=2,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1
            ))
        ])
        boost.fit(X_tr, y_tr)

    proba_b = boost.predict_proba(X_te)[:, 1]
    pred_b = (proba_b >= THR).astype(np.uint8)

    auc_b = roc_auc_score(y_te, proba_b)
    ll_b  = log_loss(y_te, proba_b)
    f1_b  = f1_score(y_te, pred_b)
    pr_b  = average_precision_score(y_te, proba_b)

    print(f"[DEBUG] {boost_name.upper():>5}  target={args.target}  thr={THR:.2f}  predicted_positive_rate={pred_b.mean():.3f}")

    # Save pipelines (include target in filename so you don't overwrite by accident)
    import joblib
    joblib.dump(logit, outdir / f"logit_pipeline_{args.target}.joblib")
    joblib.dump(boost, outdir / f"boost_pipeline_{args.target}.joblib")

    # Warn (don’t crash) if threshold yields degenerate all-0/all-1
    if not (0.0 < pred_l.mean() < 1.0):
        warnings.warn(f"LOGIT degenerate predictions at thr={THR:.2f} (rate={pred_l.mean():.3f})")
    if not (0.0 < pred_b.mean() < 1.0):
        warnings.warn(f"{boost_name.upper()} degenerate predictions at thr={THR:.2f} (rate={pred_b.mean():.3f})")

    # 9) Plots: ROC + calibration
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    RocCurveDisplay.from_predictions(y_te, proba_l, name="LOGIT", ax=ax)
    ax.set_title(f"ROC LOGIT (AUC={auc_l:.3f})")
    fig.tight_layout()
    fig.savefig(outdir / "roc_logit.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    RocCurveDisplay.from_predictions(y_te, proba_b, name=boost_name.upper(), ax=ax)
    ax.set_title(f"ROC {boost_name.upper()} (AUC={auc_b:.3f})")
    fig.tight_layout()
    fig.savefig(outdir / "roc_boost.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    CalibrationDisplay.from_predictions(y_te, proba_l, n_bins=10, name="LOGIT", ax=ax)
    ax.set_title("Calibration: LOGIT")
    fig.tight_layout()
    fig.savefig(outdir / "calib_logit.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    CalibrationDisplay.from_predictions(y_te, proba_b, n_bins=10, name=boost_name.upper(), ax=ax)
    ax.set_title(f"Calibration: {boost_name.upper()}")
    fig.tight_layout()
    fig.savefig(outdir / "calib_boost.png")
    plt.close(fig)

    # 10) Save predictions
    test_frame = model_df.loc[X_te.index, ["game_id", "date", "hometeam", "visteam"]].copy()
    test_frame["y_true"] = y_te
    test_frame["pred_logit"] = pred_l
    test_frame["pred_boost"] = pred_b
    test_frame["p_logit"] = proba_l
    test_frame["p_boost"] = proba_b
    test_frame["thr"] = THR
    test_frame["target"] = args.target
    test_frame.to_csv(outdir / "predictions_logit.csv", index=False)
    test_frame.to_csv(outdir / "predictions_boost.csv", index=False)

    # 11) Feature importances for boosted model (if available)
    try:
        prefit = boost.named_steps["pre"]
        num_names = features_num
        cat_pipe = prefit.named_transformers_["cat"]
        cat_ohe = cat_pipe.named_steps.get("ohe", cat_pipe)
        cat_names = list(cat_ohe.get_feature_names_out(features_cat))
        all_names = num_names + cat_names

        importances = boost.named_steps["clf"].feature_importances_
        imp_df = (
            pd.DataFrame({"feature": all_names, "importance": importances})
            .sort_values("importance", ascending=False)
        )
        imp_df.to_csv(outdir / "feature_importances_boost.csv", index=False)
    except Exception as e:
        warnings.warn(f"Could not export feature importances: {e}")

    # 12) Metrics file
    lines = []
    lines.append(f"Target: {args.target}")
    lines.append(f"Weighted base (year decay): {args.weight_base}")
    lines.append(f"Fixed classification threshold: {THR:.2f}")
    lines.append("")

    lines.append("LOGISTIC REGRESSION (weighted by year):")
    lines.append(
        f"AUC={auc_l:.3f}  F1={f1_l:.3f}  PR-AUC={pr_l:.3f}  LogLoss={ll_l:.4f}  "
        f"Predicted+={pred_l.mean():.3%}"
    )
    lines.append(f"True prevalence (test): {np.mean(y_te):.3%}")
    lines.append(classification_report(y_te, pred_l, digits=3, zero_division=0))
    lines.append(f"Confusion matrix (LOGIT):\n{confusion_matrix(y_te, pred_l)}")
    lines.append("")

    lines.append(f"{boost_name.upper()} MODEL:")
    lines.append(
        f"AUC={auc_b:.3f}  F1={f1_b:.3f}  PR-AUC={pr_b:.3f}  LogLoss={ll_b:.4f}  "
        f"Predicted+={pred_b.mean():.3%}"
    )
    lines.append(f"True prevalence (test): {np.mean(y_te):.3%}")
    lines.append(classification_report(y_te, pred_b, digits=3, zero_division=0))
    lines.append(f"Confusion matrix ({boost_name.upper()}):\n{confusion_matrix(y_te, pred_b)}")

    (outdir / "model_metrics_weighted.txt").write_text("\n".join(lines), encoding="utf-8")

    print("[OK] Saved:")
    print(f"  - {outdir/'modeling_dataset_weighted.csv'}")
    print(f"  - {outdir/'model_metrics_weighted.txt'}")
    print(f"  - {outdir/'roc_logit.png'}, {outdir/'roc_boost.png'}")
    print(f"  - {outdir/'calib_logit.png'}, {outdir/'calib_boost.png'}")
    print(f"  - {outdir/'feature_importances_boost.csv'}")
    print(f"  - {outdir/'predictions_logit.csv'}, {outdir/'predictions_boost.csv'}")
    print(f"  - {outdir/f'logit_pipeline_{args.target}.joblib'}")
    print(f"  - {outdir/f'boost_pipeline_{args.target}.joblib'}")


if __name__ == "__main__":
    main()
