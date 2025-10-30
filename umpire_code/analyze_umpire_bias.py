#!/usr/bin/env python3
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT  = Path(__file__).resolve().parents[1]
FEAT  = ROOT / "pitches_features.parquet"
OUT1  = ROOT / "park_quadrant_fringe_summary.csv"
OUT2  = ROOT / "team_home_edge_delta.csv"
PLOT1 = ROOT / "fringe_home_edge_delta.png"

INCH = 1/12
FRINGE = 2*INCH
HALF_W = 0.83

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    need = ["home_team","inning","inning_topbot","plate_x","plate_z","sz_bot","sz_top",
            "is_called_strike"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in features parquet: {missing}")
    return df

def recompute_fringe(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee is_fringe and fringe_side exist."""
    if ("is_fringe" in df.columns) and ("fringe_side" in df.columns):
        return df

    # recompute distances to rulebook edges
    dx_left  = (df["plate_x"] + HALF_W).abs()
    dx_right = (df["plate_x"] - HALF_W).abs()
    dz_bot   = (df["plate_z"] - df["sz_bot"]).abs()
    dz_top   = (df["plate_z"] - df["sz_top"]).abs()

    edge_dist = pd.concat([dx_left, dx_right, dz_bot, dz_top], axis=1).min(axis=1)
    df["is_fringe"] = edge_dist <= FRINGE

    # nearest side
    arr = np.vstack([dx_left.to_numpy(), dx_right.to_numpy(),
                     dz_bot.to_numpy(), dz_top.to_numpy()]).T
    idx = np.argmin(arr, axis=1)
    sides = np.array(["left","right","bottom","top"])
    df["fringe_side"] = sides[idx]
    return df

def main():
    if not FEAT.exists():
        raise SystemExit(f"Missing {FEAT}. Run build_strike_features.py first.")

    df = pd.read_parquet(FEAT)
    df = ensure_cols(df)

    # basic flags if missing
    if "first_inning" not in df.columns:
        df["first_inning"] = (df["inning"] == 1).astype(int)
    if "is_top" not in df.columns:
        df["is_top"] = df["inning_topbot"].astype(str).str.upper().eq("TOP").astype(int)

    # (Re)compute fringe + side if not present
    df = recompute_fringe(df)

    edge = df[df["is_fringe"]].copy()
    edge["is_home_pitch"] = (edge["is_top"] == 0)

    # Park × side × quadrant × first_inning summaries (quadrant optional)
    if "quadrant" not in edge.columns:
        edge["quadrant"] = "UNK"

    grp = edge.groupby(
        ["home_team","fringe_side","quadrant","first_inning"],
        dropna=False
    ).agg(
        n=("is_called_strike","size"),
        csr=("is_called_strike","mean"),
        exp=("p_hat","mean") if "p_hat" in edge.columns else ("is_called_strike","mean")
    ).reset_index()

    # If p_hat missing, exp==csr → ocs=0; we still produce tables
    if "p_hat" not in edge.columns:
        print("p_hat missing in features; OCS will be zeros.")
    grp["ocs"] = grp["csr"] - grp["exp"]
    grp.to_csv(OUT1, index=False)
    print(f"[ok] wrote {OUT1}")

    # Home vs visitor Δ(OCS) on fringes
    g2 = edge.groupby(
        ["home_team","fringe_side","first_inning","is_home_pitch"],
        dropna=False
    ).agg(
        csr=("is_called_strike","mean"),
        exp=("p_hat","mean") if "p_hat" in edge.columns else ("is_called_strike","mean"),
        n=("is_called_strike","size")
    ).reset_index()

    wide = g2.pivot(index=["home_team","fringe_side","first_inning"],
                    columns="is_home_pitch", values=["csr","exp","n"]).fillna(0)

    # columns: (metric, False) visitor, (metric, True) home
    home_csr = wide[("csr", True)]; away_csr = wide[("csr", False)]
    home_exp = wide[("exp", True)]; away_exp = wide[("exp", False)]
    home_edge_delta = (home_csr - home_exp) - (away_csr - away_exp)

    out = home_edge_delta.rename("home_edge_delta").reset_index()
    out.to_csv(OUT2, index=False)
    print(f"[ok] wrote {OUT2}")

    # Plot: top absolute deltas, first inning, left/right edges
    f = out[(out["first_inning"]==1) & (out["fringe_side"].isin(["left","right"]))].copy()
    if f.empty:
        print("No first-inning fringe rows to plot; skipping figure.")
        return
    f["abs_delta"] = f["home_edge_delta"].abs()
    top = f.sort_values("abs_delta", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9,5), dpi=140)
    ax.barh(top["home_team"]+" "+top["fringe_side"], top["home_edge_delta"])
    ax.axvline(0, linewidth=1, linestyle="--")
    ax.set_title("First-Inning Over-Expected Called Strikes: Home − Visitor (Fringes)")
    ax.set_xlabel("Δ(OCS)")
    fig.tight_layout()
    fig.savefig(PLOT1)
    print(f"[ok] saved {PLOT1}")

if __name__ == "__main__":
    main()
