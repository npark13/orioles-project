#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze 2024 holdout predictions and generate a confusion matrix heatmap.

Uses:
  - predictions_2024_full_comparison.csv

Columns expected:
  game_id,date,hometeam,visteam,y_true,
  pred_logit,pred_boost,p_logit,p_boost,thr_logit,thr_boost
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score


def main():
    parser = argparse.ArgumentParser(
        description="Generate confusion matrix heatmap for 2024 XGBoost predictions."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="predictions_2024_full_comparison.csv",
        help="Path to predictions CSV (default: predictions_2024_full_comparison.csv)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="confusion_matrix_2024_boost.png",
        help="Output PNG filename for the heatmap.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    # 1) Load data
    df = pd.read_csv(csv_path)

    # Sanity checks
    required_cols = {"y_true", "pred_boost"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in CSV: {missing}")

    y_true = df["y_true"].astype(int).values
    y_pred = df["pred_boost"].astype(int).values

    # 2) Compute confusion matrix & accuracy
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true, y_pred)

    print("=== XGBoost model performance on 2024 ===")
    print(f"Total games: {len(y_true)}")
    print(f"Accuracy : {acc:.4f}")
    print(f"TN (correct NRFI): {tn}")
    print(f"FP (pred YRFI, actual NRFI): {fp}")
    print(f"FN (pred NRFI, actual YRFI): {fn}")
    print(f"TP (correct YRFI): {tp}")
    print()
    print("Confusion matrix (rows=Actual [NRFI, YRFI], cols=Predicted [NRFI, YRFI]):")
    print(cm)

    # 3) Plot heatmap using matplotlib only
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm)

    # Tick labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: NRFI", "Pred: YRFI"])
    ax.set_yticklabels(["Actual: NRFI", "Actual: YRFI"])

    # Annotate cells with counts
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=12,
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — XGBoost on 2024 Holdout")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)

    print(f"\nSaved confusion matrix heatmap to: {args.out}")


if __name__ == "__main__":
    main()
