#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-ids", type=str, required=True,
                    help="CSV file with a column 'game_id' listing games to predict")
    ap.add_argument("--model-dataset", type=str, default="modeling_dataset_weighted.csv",
                    help="CSV with all features used for modeling")
    ap.add_argument("--logit-pipeline", type=str, default="logit_pipeline.joblib")
    ap.add_argument("--boost-pipeline", type=str, default="boost_pipeline.joblib")
    ap.add_argument("--out", type=str, default="predictions_for_games.csv")
    args = ap.parse_args()

    # Load model dataset
    model_df = pd.read_csv(args.model_dataset)
    
    # Load target game IDs
    game_ids_df = pd.read_csv(args.game_ids)
    if "game_id" not in game_ids_df.columns:
        raise SystemExit("Input CSV must have a 'game_id' column")

    game_ids = game_ids_df["game_id"].unique()

    # Subset features for these game IDs
    df = model_df[model_df["game_id"].isin(game_ids)].copy()
    if df.empty:
        raise SystemExit("No matching game IDs found in model dataset")

    # Identify feature columns used in modeling
    feature_cols = [c for c in model_df.columns if c not in ["game_id","date","hometeam","visteam","first_inning_any"]]

    X = df[feature_cols]

    # Load trained pipelines
    logit = joblib.load(args.logit_pipeline)
    boost = joblib.load(args.boost_pipeline)

    # Make predictions
    df["pred_logit_prob"] = logit.predict_proba(X)[:, 1]
    df["pred_logit"] = (df["pred_logit_prob"] >= 0.5).astype(int)

    df["pred_boost_prob"] = boost.predict_proba(X)[:, 1]
    df["pred_boost"] = (df["pred_boost_prob"] >= 0.5).astype(int)

    # Save results
    df[["game_id","hometeam","visteam","pred_logit","pred_logit_prob","pred_boost","pred_boost_prob"]].to_csv(
        args.out, index=False
    )

    print(f"[OK] Predictions saved to {args.out}")

if __name__ == "__main__":
    main()