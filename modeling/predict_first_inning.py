import pandas as pd
import joblib
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Predict first-inning scoring probabilities using logistic regression and XGBoost pipelines."
    )
    parser.add_argument("--game-ids", required=True, help="CSV containing game_id column")
    parser.add_argument("--model-dataset", required=True, help="Full modeling dataset with features")
    parser.add_argument("--logit-pipeline", required=True, help="Path to saved logistic regression pipeline (.joblib)")
    parser.add_argument("--boost-pipeline", required=True, help="Path to saved XGBoost pipeline (.joblib)")
    parser.add_argument("--out", required=True, help="Output CSV file for predictions")
    args = parser.parse_args()

    try:
        # Load input files
        print("Loading input data...")
        game_ids_df = pd.read_csv(args.game_ids)
        model_df = pd.read_csv(args.model_dataset)

        if "game_id" not in game_ids_df.columns:
            raise ValueError("`game_ids.csv` must contain a 'game_id' column.")
        if "game_id" not in model_df.columns:
            raise ValueError("Model dataset must contain a 'game_id' column.")

        # Merge on game_id to get full feature rows
        print("Merging datasets...")
        merged_df = pd.merge(game_ids_df, model_df, on="game_id", how="left")

        if merged_df.isnull().all(axis=1).any():
            print("Warning: Some game IDs did not match any rows in the model dataset.", file=sys.stderr)

        # Load saved pipelines
        print("Loading saved model pipelines...")
        logit = joblib.load(args.logit_pipeline)
        boost = joblib.load(args.boost_pipeline)

<<<<<<< HEAD
        # Identify required features
        # Pipelines built with ColumnTransformer store the names in `feature_names_in_`
=======
        # 4️⃣ Identify required features
>>>>>>> 1cb2a45 (got sample data for nrfi)
        try:
            feature_cols = logit.named_steps["columntransformer"].feature_names_in_
        except Exception:
            feature_cols = model_df.columns.drop("game_id")

        missing_cols = set(feature_cols) - set(merged_df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")

        X = merged_df[feature_cols]

        # Make predictions
        print("Making predictions...")
        merged_df["logit_prob_first_inning_score"] = logit.predict_proba(X)[:, 1]
        merged_df["boost_prob_first_inning_score"] = boost.predict_proba(X)[:, 1]

<<<<<<< HEAD
        # Output results
        output_cols = ["game_id", "logit_prob_first_inning_score", "boost_prob_first_inning_score"]
        merged_df[output_cols].to_csv(args.out, index=False)
        print(f"Predictions saved to {args.out}")
=======
        # 6️⃣ Filter out extreme probabilities (>0.95 or <0.05)
        mask = (
            merged_df["logit_prob_first_inning_score"].between(0.05, 0.95) &
            merged_df["boost_prob_first_inning_score"].between(0.05, 0.95)
        )
        filtered_df = merged_df[mask]
        print(f"Filtered out {len(merged_df) - len(filtered_df)} games with extreme probabilities.")

        # 7️⃣ Output results
        output_cols = ["game_id", "logit_prob_first_inning_score", "boost_prob_first_inning_score"]
        filtered_df[output_cols].to_csv(args.out, index=False)
        print(f"✅ Filtered predictions saved to {args.out}")
>>>>>>> 1cb2a45 (got sample data for nrfi)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
