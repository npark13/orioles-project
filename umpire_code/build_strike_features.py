#!/usr/bin/env python3
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "statcast_pitches_raw_2018_2024.parquet"
FEAT = ROOT / "pitches_features.parquet"

INCH = 1/12
FRINGE = 2*INCH
HALF_W = 0.83

# ---------- utilities ----------
def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist; add if missing and print a warning."""
    required = [
        # numeric
        "plate_x","plate_z","sz_bot","sz_top","inning",
        # categorical
        "inning_topbot","pitch_type","p_throws","stands",
        "balls","strikes","description","type"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[warn] adding missing columns: {missing}")
        for c in missing:
            # choose sensible default dtype
            if c in ["plate_x","plate_z","sz_bot","sz_top","inning","balls","strikes"]:
                df[c] = np.nan
            else:
                df[c] = np.nan
    return df

def coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---------- labeling & features ----------
def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # keep called balls/strikes only
    # tolerate whichever columns exist
    desc = df["description"].astype(str) if "description" in df.columns else pd.Series("", index=df.index)
    typ  = df["type"].astype(str) if "type" in df.columns else pd.Series("", index=df.index)
    is_called = desc.isin(["called_strike","ball"]) | typ.isin(["S","B"])
    df = df[is_called].copy()
    print(f"[build] called-only rows: {len(df):,}")

    # ensure core columns present
    df = ensure_columns(df)

    # outcome
    df["is_called_strike"] = (df["description"] == "called_strike") | (df["type"] == "S")

    # numeric coercion
    df = coerce_numeric(df, ["plate_x","plate_z","sz_bot","sz_top","inning","balls","strikes"])
    df = df.dropna(subset=["plate_x","plate_z","sz_bot","sz_top"])

    # zone geometry
    inside_h = df["plate_x"].between(-HALF_W, HALF_W, inclusive="both")
    inside_v = df["plate_z"].between(df["sz_bot"], df["sz_top"], inclusive="both")
    df["is_inside_zone"] = inside_h & inside_v

    df["dx_left"]  = (df["plate_x"] + HALF_W).abs()
    df["dx_right"] = (df["plate_x"] - HALF_W).abs()
    df["dz_bot"]   = (df["plate_z"] - df["sz_bot"]).abs()
    df["dz_top"]   = (df["plate_z"] - df["sz_top"]).abs()
    df["edge_dist"] = df[["dx_left","dx_right","dz_bot","dz_top"]].min(axis=1)
    df["is_fringe"] = df["edge_dist"] <= FRINGE

    # inning orientation
    topbot = df["inning_topbot"].astype(str).str.upper().fillna("TOP")
    df["first_inning"] = (df["inning"] == 1).astype(int)
    df["is_top"] = (topbot == "TOP").astype(int)

    # quadrant
    def quad(r):
        if not r["is_inside_zone"]:
            return "OZ"
        zmid = (r["sz_bot"] + r["sz_top"]) / 2
        vert = "Top" if r["plate_z"] >= zmid else "Bottom"
        horiz = "Right" if r["plate_x"] >= 0 else "Left"
        return f"{vert}-{horiz}"
    df["quadrant"] = df.apply(quad, axis=1)

    # categorical hygiene
    for c in ["pitch_type","p_throws","stands"]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("UNK")
        else:
            df[c] = "UNK"
    # count string
    df["count"] = df["balls"].fillna(0).astype(int).astype(str) + "-" + df["strikes"].fillna(0).astype(int).astype(str)

    print(f"[build] rows after cleaning: {len(df):,}")
    return df

def fit_expected_model(df: pd.DataFrame) -> pd.Series:
    # define desired feature lists
    Xnum_cols = ["plate_x","plate_z","sz_bot","sz_top","first_inning"]
    Xcat_cols = ["count","pitch_type","stands","p_throws","quadrant"]

    # only use columns that actually exist
    Xnum_cols_avail = [c for c in Xnum_cols if c in df.columns]
    Xcat_cols_avail = [c for c in Xcat_cols if c in df.columns]
    missing_any = [c for c in (Xnum_cols+Xcat_cols) if c not in (Xnum_cols_avail+Xcat_cols_avail)]
    if missing_any:
        print(f"[warn] expected-but-missing feature cols will be skipped: {missing_any}")

    Xnum = df[Xnum_cols_avail]
    Xcat = df[Xcat_cols_avail].fillna("UNK")
    y = df["is_called_strike"].astype(int)

    pre = ColumnTransformer([
        ("num", "passthrough", Xnum_cols_avail),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), Xcat_cols_avail),
    ])
    clf = LogisticRegression(max_iter=300)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X = pd.concat([Xnum, Xcat], axis=1)
    print("[build] fitting logistic regression on", X.shape[0], "rows and",
          len(Xnum_cols_avail), "numeric +", len(Xcat_cols_avail), "categorical feature groups")
    pipe.fit(X, y)
    p_hat = pipe.predict_proba(X)[:, 1]
    return pd.Series(p_hat, index=df.index, name="p_hat")

# ---------- main ----------
def main():
    print("[build] looking for:", RAW)
    if not RAW.exists():
        raise SystemExit(f"Missing {RAW}. Run statcast_scrape.py first.")

    df = pd.read_parquet(RAW)
    print(f"[build] start rows: {len(df):,}")
    if df.empty:
        raise SystemExit("[build] raw file has 0 rows; re-run the scraper with a wider date range.")

    df = ensure_columns(df)
    df = add_labels(df)
    if df.empty:
        raise SystemExit("[build] no called balls/strikes after cleaning; widen the date range.")

    df["p_hat"] = fit_expected_model(df)
    df.to_parquet(FEAT, index=False)
    print(f"[OK] wrote {FEAT}  rows={len(df):,}")

if __name__ == "__main__":
    main()
