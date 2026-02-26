"""
Evaluation script: produces comprehensive metrics, confusion matrix,
and evaluation report.

Usage:
    python scripts/evaluate.py [--data-dir data/] [--models-dir models/]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, top_k_accuracy_score,
    classification_report, confusion_matrix,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import CROP_LABELS
from src.data.features import load_pipeline, get_model_features
from src.models.classifier import CropClassifier
from src.models.yield_model import YieldPredictor
from src.explain.shap_explain import SHAPExplainer


def evaluate(data_dir: str = "data", models_dir: str = "models") -> dict:
    """Run full evaluation suite."""
    data_dir = Path(data_dir)
    models_dir = Path(models_dir)

    print("=" * 60)
    print("EVALUATION SUITE")
    print("=" * 60)

    # Load data
    df = pd.read_parquet(data_dir / "processed" / "crop_recommendation_clean.parquet")

    # Load models
    clf = CropClassifier.load(str(models_dir / "crop_classifier.joblib"))
    yield_model = YieldPredictor.load(str(models_dir / "yield_predictor.joblib"))
    pipeline = load_pipeline(str(models_dir / "feature_pipeline.joblib"))

    # Prepare features
    df_transformed = pipeline.transform(df)
    feature_cols = get_model_features(df_transformed)
    X = df_transformed[feature_cols].astype(float).fillna(0).values

    # Encode target
    le = LabelEncoder()
    le.fit(CROP_LABELS)
    y = le.transform(df["target_crop"].values)
    groups = df["location_id"].values

    # Geographic split (same as training)
    unique_locs = np.unique(groups)
    n_test_locs = max(1, len(unique_locs) // 5)
    np.random.seed(42)
    test_locs = set(np.random.choice(unique_locs, size=n_test_locs, replace=False))
    test_mask = np.isin(groups, list(test_locs))

    X_test = X[test_mask]
    y_test = y[test_mask]
    y_yield_test = df["yield_kg_ha"].values[test_mask]

    # Classification metrics
    proba = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)

    top1_acc = accuracy_score(y_test, y_pred)
    top3_acc = top_k_accuracy_score(y_test, proba, k=3, labels=range(len(CROP_LABELS)))

    print(f"\nClassification Metrics (geo-held-out test set, n={len(X_test)}):")
    print(f"  Top-1 Accuracy: {top1_acc:.4f}")
    print(f"  Top-3 Accuracy: {top3_acc:.4f}")

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(CROP_LABELS)), zero_division=0
    )
    per_class = {}
    for i, crop in enumerate(CROP_LABELS):
        if support[i] > 0:
            per_class[crop] = {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
    print(f"\nPer-class metrics for {len(per_class)} classes with test samples")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=range(len(CROP_LABELS)))

    # Yield MAE
    yield_eval = yield_model.evaluate(
        X_test, y_yield_test,
        crop_labels=le.inverse_transform(y_test)
    )
    print(f"\nYield MAE: {yield_eval['overall_mae']}")
    if yield_eval.get("per_crop_mae"):
        print("Per-crop MAE:")
        for crop, mae in sorted(yield_eval["per_crop_mae"].items()):
            print(f"  {crop}: {mae:.0f} kg/ha")

    # SHAP global importance
    print("\nComputing SHAP global feature importance...")
    explainer = SHAPExplainer(clf.model, feature_names=clf.feature_names)
    global_importance = explainer.global_feature_importance(X_test)
    print("Top 10 features:")
    for feat, imp in list(global_importance.items())[:10]:
        print(f"  {feat}: {imp:.4f}")

    # Compile report
    report = {
        "evaluation_date": pd.Timestamp.now().isoformat(),
        "test_set_size": int(len(X_test)),
        "n_test_locations": int(n_test_locs),
        "classification": {
            "top1_accuracy": round(float(top1_acc), 4),
            "top3_accuracy": round(float(top3_acc), 4),
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
        },
        "yield": yield_eval,
        "shap_global_importance": global_importance,
        "target_met": {
            "top3_accuracy_ge_075": top3_acc >= 0.75,
            "remediation": None if top3_acc >= 0.75 else
                "Consider: (1) more training data, (2) hyperparameter tuning, "
                "(3) additional features (micronutrients, irrigation, OC)",
        },
    }

    report_path = models_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nEvaluation report saved to {report_path}")

    # Also create markdown report
    md_path = Path("docs") / "evaluation_report.md"
    md_path.parent.mkdir(exist_ok=True)
    with open(md_path, "w") as f:
        f.write("# Crop Recommendation — Evaluation Report\n\n")
        f.write(f"**Date:** {report['evaluation_date']}\n\n")
        f.write(f"**Test set:** {report['test_set_size']} samples from {report['n_test_locations']} held-out locations\n\n")
        f.write("## Classification Metrics\n\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Top-1 Accuracy | {top1_acc:.4f} |\n")
        f.write(f"| Top-3 Accuracy | {top3_acc:.4f} |\n\n")
        f.write("## Per-Crop Performance\n\n")
        f.write("| Crop | Precision | Recall | F1 | Support |\n")
        f.write("|------|-----------|--------|----|---------|\n")
        for crop, m in sorted(per_class.items()):
            f.write(f"| {crop} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |\n")
        f.write(f"\n## Yield MAE\n\n- Overall: {yield_eval['overall_mae']} kg/ha\n\n")
        if yield_eval.get("per_crop_mae"):
            f.write("| Crop | MAE (kg/ha) |\n|------|-------------|\n")
            for crop, mae in sorted(yield_eval["per_crop_mae"].items()):
                f.write(f"| {crop} | {mae:.0f} |\n")
        f.write("\n## SHAP Feature Importance (Top 10)\n\n")
        f.write("| Feature | Importance |\n|---------|------------|\n")
        for feat, imp in list(global_importance.items())[:10]:
            f.write(f"| {feat} | {imp:.4f} |\n")
        if report["target_met"]["top3_accuracy_ge_075"]:
            f.write("\n## ✓ Target Met\n\nTop-3 accuracy ≥ 0.75 achieved.\n")
        else:
            f.write(f"\n## ⚠ Target Not Met\n\n{report['target_met']['remediation']}\n")
    print(f"Markdown report saved to {md_path}")

    print("=" * 60)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate crop recommendation models")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()
    evaluate(args.data_dir, args.models_dir)
