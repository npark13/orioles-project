#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weighted Logistic + Boosted Model for 1st-inning home scoring.

Train on all seasons up to and including 2023,
test on 2024 only.

Outputs (to --outdir, default "."):
- modeling_dataset_weighted.csv
- model_metrics_weighted.txt
- roc_logit.png, roc_boost.png
- calib_logit.png, calib_boost.png
- feature_importances_boost.csv
- predictions_logit.csv
- predictions_boost.csv
- logit_pipeline.joblib
- boost_pipeline.joblib
"""

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

def tz_from_lon(lon: float) -> int:
    if lon <= -125: return -8
    if lon <= -115: return -7
    if lon <= -100: return -6
    if lon <= -85:  return -5
    return -4

def load_games_range(games_root: Path, start: int, end: int) -> pd.DataFrame:
    frames = []
    for y in range(start, end+1):
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
        keep = [c for c in ["game_id","date","hometeam","visteam","site","number"] if c in g.columns]
        frames.append(g[keep])
    if not frames:
        raise SystemExit(f"No games.csv found under {games_root} for {start}-{end}")
    return pd.concat(frames, ignore_index=True).dropna(subset=["game_id","date","hometeam","visteam"])

def attach_travel_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["visteam","date","number"]).copy()
    last_loc, last_time = {}, {}
    travel_km, tz_dir, rest_hours = [], [], []

    for _, r in df.iterrows():
        home, vis = r["hometeam"], r["visteam"]
        if home not in TEAM_COORDS or vis not in TEAM_COORDS:
            travel_km.append(np.nan); tz_dir.append("unknown"); rest_hours.append(np.nan); continue
        dest_lat, dest_lon = TEAM_COORDS[home]

        if vis in last_loc:
            prev_lat, prev_lon = last_loc[vis]
            dkm = haversine_km(prev_lat, prev_lon, dest_lat, dest_lon)
            tz_prev = tz_from_lon(prev_lon)
        else:
            dkm = np.nan
            tz_prev = tz_from_lon(dest_lon)

        tz_now = tz_from_lon(dest_lon)
        if tz_now > tz_prev: dirlab = "eastbound"
        elif tz_now < tz_prev: dirlab = "westbound"
        else: dirlab = "same_zone"

        if vis in last_time:
            rh = (r["date"] - last_time[vis]).total_seconds()/3600.0
        else:
            rh = np.nan

        travel_km.append(dkm)
        tz_dir.append(dirlab)
        rest_hours.append(rh)

        last_loc[vis]  = (dest_lat, dest_lon)
        last_time[vis] = r["date"]

    df["visitor_travel_km"] = pd.Series(travel_km, index=df.index)
    df["tz_dir"] = pd.Series(tz_dir, index=df.index)
    df["rest_hours_since_last"] = pd.Series(rest_hours, index=df.index)
    return df

def pick_bias_column(cols: list[str]) -> str | None:
    for c in ["umpire_bias_index","home_bias_index","bias_index","ocs_bias"]:
        if c in cols:
            return c
    return None

def year_weights(dates: pd.Series, base: float = 0.90) -> np.ndarray:
    years = pd.to_datetime(dates).dt.year
    ymax = years.max()
    return np.power(base, ymax - years).astype(float)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def topk_mask_by_rank(y_true, proba, lo=0.25, hi=0.35):
    """
    Select exactly k positives by rank where
    k = round(clamp(prevalence, lo, hi) * n), then 1..n-1 clamp.
    Returns (mask[int{0,1}], cutoff_prob, k, rate_target, rate_true).
    """
    proba = np.asarray(proba, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    n = len(proba)
    if n == 0:
        return np.zeros(0, dtype=np.uint8), 0.5, 0, 0.0, 0.0

    prevalence = float(y_true.mean()) if y_true.size else 0.3
    target_rate = clamp(prevalence, lo, hi)
    k = int(round(target_rate * n))
    k = max(1, min(n - 1, k))  # avoid degenerate all/none

    order = np.argsort(-proba, kind="mergesort")
    idx = order[:k]

    mask = np.zeros(n, dtype=np.uint8)
    mask[idx] = 1

    cutoff = float(proba[idx[-1]]) if k > 0 else 0.5
    return mask, cutoff, k, target_rate, float(mask.mean())


# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2013)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--games-root", type=str, default="data/out")
    ap.add_argument("--per-inning", type=str, required=True,
                    help="CSV with per-inning runs; requires columns: game_id, home_R1")
    ap.add_argument("--results", type=str, default="results_by_game.csv")
    ap.add_argument("--umpire-index", type=str, default=None)
    ap.add_argument("--home-era", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=".")
    ap.add_argument("--weight-base", type=float, default=0.90,
                    help="Year decay base; lower = more recent emphasis.")
    args = ap.parse_args()

    root = Path(".").resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) load games and attach travel features
    games = load_games_range(root / args.games_root, args.start, args.end)
    games = attach_travel_features(games)

    # 2) target from per-inning
    per_inning = pd.read_csv(
        root / args.per_inning,
        engine="python",
        on_bad_lines="warn"  # skip malformed rows, warn instead of crashing
    )

    per_inning.columns = [c.lower() for c in per_inning.columns]

    if "game_id" not in per_inning.columns:
        raise SystemExit(
            "per-inning / rolling_avg file is missing 'game_id'.\n"
            f"Columns present: {per_inning.columns.tolist()}"
        )

    if "home_r1" not in per_inning.columns:
        if "home_first_inning_runs" in per_inning.columns:
            per_inning = per_inning.rename(columns={
                "home_first_inning_runs": "home_r1",
                "home_era": "home_starter_era",
            })
            print("[DEBUG] Renamed home_first_inning_runs → home_r1")
        else:
            raise SystemExit(
                "per-inning / rolling_avg file must include either 'home_r1' "
                "or 'home_first_inning_runs'.\n"
                f"Got columns: {per_inning.columns.tolist()}"
            )

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

    df = games.merge(
        per_inning[["game_id", "home_r1"] + present_extra],
        on="game_id",
        how="left"
    )

    df["home_scores_1st"] = (df["home_r1"].fillna(0) > 0).astype(int)

    for c in present_extra:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 3) optional: umpire bias
    if args.umpire_index:
        ump_fp = root / args.umpire_index
        if ump_fp.exists():
            ump = pd.read_csv(ump_fp)
            ump.columns = [c.lower() for c in ump.columns]
            bcol = pick_bias_column(list(ump.columns))
            if bcol and "game_id" in ump.columns:
                df = df.merge(
                    ump[["game_id", bcol]].rename(columns={bcol: "umpire_bias_index"}),
                    on="game_id",
                    how="left"
                )
            else:
                warnings.warn("Umpire index missing (game_id or bias column). Skipping.")
        else:
            warnings.warn(f"Umpire index not found: {ump_fp}. Skipping.")
    if "umpire_bias_index" not in df.columns:
        df["umpire_bias_index"] = 0.0

    # 4) optional: home ERA (game-level or seasonal)
    if args.home_era:
        era_fp = root / args.home_era
        if era_fp.exists():
            era = pd.read_csv(era_fp)
            era.columns = [c.lower() for c in era.columns]
            if {"game_id", "home_starter_era"}.issubset(era.columns):
                df = df.merge(era[["game_id", "home_starter_era"]], on="game_id", how="left")
            elif {"hometeam", "season", "home_era"}.issubset(era.columns):
                df["season"] = df["date"].dt.year
                df = (
                    df.merge(
                        era[["hometeam", "season", "home_era"]],
                        on=["hometeam", "season"],
                        how="left"
                    )
                    .rename(columns={"home_era": "home_starter_era"})
                )
            else:
                warnings.warn("home-era file format not recognized.")
        else:
            warnings.warn(f"home-era file not found: {era_fp}.")
    if "home_starter_era" not in df.columns:
        df["home_starter_era"] = np.nan

    # Optional: merge recent form
    recent_fp = root / "data/out/home_recent_form.csv"
    if recent_fp.exists():
        recent = pd.read_csv(recent_fp)
        recent.columns = [c.lower() for c in recent.columns]
        if {"game_id", "home_recent_form"}.issubset(recent.columns):
            df = df.merge(
                recent[["game_id", "home_recent_form"]],
                on="game_id",
                how="left"
            )
            print("✓ Added recent form feature")

    # 5) clean & feature table
    df = df.dropna(subset=["home_r1"])
    df["visitor_travel_km"] = df["visitor_travel_km"].fillna(0).clip(0, 6000)
    df["rest_hours_since_last"] = df["rest_hours_since_last"].fillna(48).clip(0, 168)
    df["umpire_bias_index"] = df["umpire_bias_index"].fillna(0).clip(-1, 1)
    df["season"] = df["date"].dt.year

    features_num = [
        "visitor_travel_km",
        "rest_hours_since_last",
        "umpire_bias_index",
        "home_starter_era",
        "home_recent_form",
    ]
    if "home_starter_era" in df.columns and df["home_starter_era"].isna().all():
        features_num.remove("home_starter_era")

    for extra in [
        "vis_era",
        "home_travel",
        "vis_travel",
        "home_avg_prev",
        "away_avg_prev",
        "home_obp",
        "away_obp",
    ]:
        if extra in df.columns and extra not in features_num:
            features_num.append(extra)

    features_cat = ["tz_dir", "hometeam"]
    keep_cols = list(
        dict.fromkeys(
            ["game_id", "date", "hometeam", "visteam", "home_scores_1st"]
            + features_num
            + features_cat
        )
    )
    model_df = df[keep_cols].copy().loc[:, ~df[keep_cols].columns.duplicated()]
    model_df.to_csv(outdir / "modeling_dataset_weighted.csv", index=False)

    # 6) explicit year-based split: train ≤ 2023, test = 2024
    model_df["year"] = model_df["date"].dt.year
    TRAIN_END = 2023
    TEST_YEAR = 2024

    train_idx = model_df["year"] <= TRAIN_END
    test_idx = model_df["year"] == TEST_YEAR

    if test_idx.sum() == 0:
        raise SystemExit(
            f"No rows found for TEST_YEAR={TEST_YEAR}. "
            "Make sure your --end includes 2024 and per-inning file has 2024 games."
        )

    X = model_df[features_num + features_cat]
    y = model_df["home_scores_1st"].astype(int).values

    X_tr, X_te = X.loc[train_idx], X.loc[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    w_tr = year_weights(model_df.loc[train_idx, "date"], base=args.weight_base)
    w_te = np.ones_like(y_te, dtype=float)

    # 7) preprocessing (impute → encode/scale)
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
        transformers=[("num", num_tf, features_num),
                      ("cat", cat_tf, features_cat)],
        remainder="drop",
        sparse_threshold=0,
    )

    # 8) weighted logistic regression
    logit = Pipeline(steps=[
        ("pre", pre),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs"
        )),
    ])
    logit.fit(X_tr, y_tr, clf__sample_weight=w_tr)
    proba_l = logit.predict_proba(X_te)[:, 1]

    pred_l, cut_l, k_l, tgt_rate_l, got_rate_l = topk_mask_by_rank(
        y_te, proba_l, lo=0.25, hi=0.35
    )
    print(
        f"[DEBUG] LOGIT  target_rate={tgt_rate_l:.3f}  k={k_l}  "
        f"got_rate={got_rate_l:.3f}  cutoff={cut_l:.6f}"
    )
    assert 0.0 < got_rate_l < 1.0, f"LOGIT degenerate got_rate={got_rate_l:.3f}"

    auc_l = roc_auc_score(y_te, proba_l)
    ll_l  = log_loss(y_te, proba_l)
    f1_l  = f1_score(y_te, pred_l)
    pr_l  = average_precision_score(y_te, proba_l)

    # 9) boosted model (xgboost > lightgbm > random forest fallback)
    boost_name = "xgb" if XGB_OK else ("lgbm" if LGB_OK else "rf")
    if XGB_OK:
        boost = Pipeline(steps=[
            ("pre", pre),
            ("clf", XGBClassifier(
                n_estimators=600, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                objective="binary:logistic", eval_metric="logloss",
                random_state=42, n_jobs=-1
            )),
        ])
        boost.fit(X_tr, y_tr, clf__sample_weight=w_tr)
    elif LGB_OK:
        boost = Pipeline(steps=[
            ("pre", pre),
            ("clf", LGBMClassifier(
                n_estimators=800, max_depth=-1, num_leaves=31,
                learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, random_state=42, n_jobs=-1
            )),
        ])
        boost.fit(X_tr, y_tr, clf__sample_weight=w_tr)
    else:
        warnings.warn("XGBoost/LightGBM not available; falling back to RandomForest.")
        boost = Pipeline(steps=[
            ("pre", pre),
            ("clf", RandomForestClassifier(
                n_estimators=500, max_depth=None, min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1
            )),
        ])
        boost.fit(X_tr, y_tr)

    proba_b = boost.predict_proba(X_te)[:, 1]
    pred_b, cut_b, k_b, tgt_rate_b, got_rate_b = topk_mask_by_rank(
        y_te, proba_b, lo=0.25, hi=0.35
    )

    import joblib
    joblib.dump(logit, outdir / "logit_pipeline.joblib")
    joblib.dump(boost, outdir / "boost_pipeline.joblib")

    print(
        f"[DEBUG] {boost_name.upper():>5}  target_rate={tgt_rate_b:.3f}  "
        f"k={k_b}  got_rate={got_rate_b:.3f}  cutoff={cut_b:.6f}"
    )
    assert 0.0 < got_rate_b < 1.0, f"{boost_name.upper()} degenerate got_rate={got_rate_b:.3f}"

    auc_b = roc_auc_score(y_te, proba_b)
    ll_b  = log_loss(y_te, proba_b)
    f1_b  = f1_score(y_te, pred_b)
    pr_b  = average_precision_score(y_te, proba_b)

    # 10) plots: ROC + calibration
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    RocCurveDisplay.from_predictions(y_te, proba_l, name="LOGIT", ax=ax)
    ax.set_title(f"ROC (AUC={auc_l:.3f}) - Test Year {TEST_YEAR}")
    fig.tight_layout()
    fig.savefig(outdir / "roc_logit.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    RocCurveDisplay.from_predictions(y_te, proba_b, name=boost_name.upper(), ax=ax)
    ax.set_title(f"ROC (AUC={auc_b:.3f}) - Test Year {TEST_YEAR}")
    fig.tight_layout()
    fig.savefig(outdir / "roc_boost.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    CalibrationDisplay.from_predictions(y_te, proba_l, n_bins=10, name="LOGIT", ax=ax)
    ax.set_title(f"Calibration: LOGIT (Test {TEST_YEAR})")
    fig.tight_layout()
    fig.savefig(outdir / "calib_logit.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    CalibrationDisplay.from_predictions(
        y_te, proba_b, n_bins=10, name=boost_name.upper(), ax=ax
    )
    ax.set_title(f"Calibration: {boost_name.upper()} (Test {TEST_YEAR})")
    fig.tight_layout()
    fig.savefig(outdir / "calib_boost.png")
    plt.close(fig)

    # 11) save predictions (2024 only)
    test_frame = model_df.loc[X_te.index, ["game_id", "date", "hometeam", "visteam"]].copy()
    test_frame["y_true"]      = y_te
    test_frame["pred_logit"]  = pred_l.astype(np.uint8)
    test_frame["pred_boost"]  = pred_b.astype(np.uint8)
    test_frame["p_logit"]     = proba_l
    test_frame["p_boost"]     = proba_b
    test_frame["thr_logit"]   = cut_l
    test_frame["thr_boost"]   = cut_b
    test_frame.to_csv(outdir / "predictions_2024_from_pre2023_logit.csv", index=False)
    test_frame.to_csv(outdir / "predictions_2024_from_pre2023_boost.csv", index=False)

    # 12) feature importances for boosted model
    try:
        prefit = boost.named_steps["pre"]
        num_names = features_num
        cat_pipe = prefit.named_transformers_["cat"]
        try:
            cat_ohe = cat_pipe.named_steps["ohe"]
        except Exception:
            cat_ohe = cat_pipe
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

    # 13) metrics file
    lines = []
    lines.append(f"Weighted base (year decay): {args.weight_base}")
    lines.append(f"Train years: ≤ {TRAIN_END}")
    lines.append(f"Test year: {TEST_YEAR}")
    lines.append("")
    lines.append("LOGISTIC REGRESSION (weighted by year):")
    lines.append(
        f"AUC={auc_l:.3f}  F1={f1_l:.3f}  PR-AUC={pr_l:.3f}  LogLoss={ll_l:.4f}  "
        f"TopK: k={k_l} ({got_rate_l:.3%})  target≈{tgt_rate_l:.3%}  cutoff={cut_l:.6f}"
    )
    lines.append(f"True prevalence (test {TEST_YEAR}): {np.mean(y_te):.3%}")
    lines.append(classification_report(y_te, pred_l, digits=3, zero_division=0))
    cm_l = confusion_matrix(y_te, pred_l)
    lines.append(f"Confusion matrix (LOGIT):\n{cm_l}")
    lines.append("")
    lines.append(f"{boost_name.upper()} MODEL:")
    lines.append(
        f"AUC={auc_b:.3f}  F1={f1_b:.3f}  PR-AUC={pr_b:.3f}  LogLoss={ll_b:.4f}  "
        f"TopK: k={k_b} ({got_rate_b:.3%})  target≈{tgt_rate_b:.3%}  cutoff={cut_b:.6f}"
    )
    lines.append(f"True prevalence (test {TEST_YEAR}): {np.mean(y_te):.3%}")
    lines.append(classification_report(y_te, pred_b, digits=3, zero_division=0))
    cm_b = confusion_matrix(y_te, pred_b)
    lines.append(f"Confusion matrix ({boost_name.upper()}):\n{cm_b}")

    (outdir / "model_metrics_weighted.txt").write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("[OK] Saved:")
    print(f"  - {outdir/'modeling_dataset_weighted.csv'}")
    print(f"  - {outdir/'model_metrics_weighted.txt'}")
    print(f"  - {outdir/'roc_logit.png'}, {outdir/'roc_boost.png'}")
    print(f"  - {outdir/'calib_logit.png'}, {outdir/'calib_boost.png'}")
    print(f"  - {outdir/'feature_importances_boost.csv'}")
    print(f"  - {outdir/'predictions_logit.csv'}, {outdir/'predictions_boost.csv'}")
    print(f"  - {outdir/'logit_pipeline.joblib'}, {outdir/'boost_pipeline.joblib'}")


if __name__ == "__main__":
    main()
