#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize 2024 prediction performance using y_true, pred_logit, pred_boost.

Expected columns in the CSV:
    y_true, pred_logit, pred_boost
(Extra columns are fine; they'll be ignored.)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score


def plot_accuracy_bar(acc_logit: float, acc_boost: float, outpath: Path) -> None:
    """Bar chart comparing accuracy of logistic vs XGBoost."""
    fig, ax = plt.subplots(figsize=(5, 4))
    models = ["Logistic", "XGBoost"]
    accuracies = [acc_logit, acc_boost]

    ax.bar(models, accuracies)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison (2024)")
    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, title: str, outpath: Path) -> None:
    """
    Plot a 2x2 confusion matrix heatmap using matplotlib only.

    cm is a 2x2 array with:
        [[TN, FP],
         [FN, TP]]
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)

    # Axis labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: 0", "Pred: 1"])
    ax.set_yticklabels(["Actual: 0", "Actual: 1"])

    # Annotate cells
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                fontsize=12,
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=str,
        default="predictions_2024_full_comparison.csv",
        help="CSV file with y_true, pred_logit, pred_boost (default: predictions_2024_full_comparison.csv)",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Directory to save plots (default: current directory)",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    # 1) Load data
    df = pd.read_csv(csv_path)

    required = {"y_true", "pred_logit", "pred_boost"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    y_true = df["y_true"].astype(int).values
    pred_logit = df["pred_logit"].astype(int).values
    pred_boost = df["pred_boost"].astype(int).values

    # 2) Accuracy
    acc_logit = accuracy_score(y_true, pred_logit)
    acc_boost = accuracy_score(y_true, pred_boost)

    print("=== 2024 Prediction Performance ===")
    print(f"Total games:           {len(y_true)}")
    print(f"Logistic accuracy:     {acc_logit:.4f}")
    print(f"XGBoost accuracy:      {acc_boost:.4f}")

    # 3) Confusion matrices
    cm_logit = confusion_matrix(y_true, pred_logit, labels=[0, 1])
    cm_boost = confusion_matrix(y_true, pred_boost, labels=[0, 1])

    print("\nLogistic confusion matrix (rows=Actual 0/1, cols=Pred 0/1):")
    print(cm_logit)
    print("\nXGBoost confusion matrix (rows=Actual 0/1, cols=Pred 0/1):")
    print(cm_boost)

    # 4) Plots
    plot_accuracy_bar(
        acc_logit,
        acc_boost,
        outdir / "accuracy_comparison_2024.png",
    )

    plot_confusion(
        cm_logit,
        "Confusion Matrix — Logistic (2024)",
        outdir / "confusion_logit_2024.png",
    )

    plot_confusion(
        cm_boost,
        "Confusion Matrix — XGBoost (2024)",
        outdir / "confusion_boost_2024.png",
    )

    print("\nSaved plots to:", outdir.resolve())
    print("  - accuracy_comparison_2024.png")
    print("  - confusion_logit_2024.png")
    print("  - confusion_boost_2024.png")


if __name__ == "__main__":
    main()
