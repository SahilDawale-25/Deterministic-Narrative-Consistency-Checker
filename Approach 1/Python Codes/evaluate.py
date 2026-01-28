# src/evaluate.py

import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def main():
    # ------------------------
    # Paths
    # ------------------------
    predictions_file = os.path.join("outputs", "results.csv")
    ground_truth_file = os.path.join("data", "train.csv")  # or test.csv

    # ------------------------
    # Check files
    # ------------------------
    if not os.path.exists(predictions_file):
        print(f"❌ Predictions file not found: {predictions_file}")
        return
    if not os.path.exists(ground_truth_file):
        print(f"❌ Ground truth file not found: {ground_truth_file}")
        return

    # ------------------------
    # Load data
    # ------------------------
    preds_df = pd.read_csv(predictions_file)
    gt_df = pd.read_csv(ground_truth_file)

    # Make sure IDs match
    merged_df = pd.merge(
        gt_df,
        preds_df,
        left_on="id",
        right_on="Story ID",
        how="inner"
    )

    if "label" in merged_df.columns:
        y_true = merged_df["label"].astype(str)
    elif "final_verdict" in merged_df.columns:
        y_true = merged_df["final_verdict"].astype(str)
    else:
        print("❌ Ground truth labels not found!")
        return

    y_pred = merged_df["Prediction"].astype(str)

    # ------------------------
    # Metrics
    # ------------------------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="consistent", zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label="consistent", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="consistent", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=["consistent", "contradict"])

    print("\n📊 Evaluation Metrics")
    print("----------------------------")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=["True Consistent", "True Contradict"], columns=["Pred Consistent", "Pred Contradict"]))

    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0, labels=["consistent", "contradict"]))

    # ------------------------
    # Optional: print accuracy per threshold
    # ------------------------
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    print("\n🔧 Threshold tuning simulation (semantic support thresholds):")
    for t in thresholds:
        y_sim = merged_df["Semantic_Score"] if "Semantic_Score" in merged_df.columns else None
        if y_sim is None:
            continue
        # Convert semantic scores to predictions
        y_pred_thresh = ["consistent" if s >= t else "contradict" for s in y_sim]
        acc_t = accuracy_score(y_true, y_pred_thresh)
        print(f"Threshold {t:.2f} → Accuracy: {acc_t:.4f}")


if __name__ == "__main__":
    main()

