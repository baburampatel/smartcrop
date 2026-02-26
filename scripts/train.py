"""
Training entry point: loads data, engineers features, trains models,
logs to MLflow, and saves best model artifacts.

Usage:
    python scripts/train.py [--data-dir data/] [--model-type xgboost] [--tune]
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import CROP_LABELS, CROP_TO_IDX
from src.data.features import build_feature_pipeline, prepare_features, save_pipeline
from src.models.baseline import evaluate_baseline
from src.models.classifier import CropClassifier, tune_hyperparameters
from src.models.yield_model import YieldPredictor

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def train(data_dir: str = "data", model_type: str = "xgboost",
          tune: bool = False, n_tune_trials: int = 30) -> dict:
    """
    Full training pipeline.
    """
    data_dir = Path(data_dir)
    processed_dir = data_dir / "processed"
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("CROP RECOMMENDATION TRAINING PIPELINE")
    print("=" * 60)

    # ---- Load data ----
    data_path = processed_dir / "crop_recommendation_clean.parquet"
    if not data_path.exists():
        print("ERROR: Cleaned data not found. Run scripts/ingest.py first.")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    print(f"[1/7] Loaded {len(df)} samples, {df['target_crop'].nunique()} crops")

    # ---- Baseline evaluation ----
    print("\n[2/7] Evaluating rule-based baseline...")
    baseline_results = evaluate_baseline(df)
    print(f"  -> Baseline Top-1: {baseline_results['top1_accuracy']:.4f}")
    print(f"  -> Baseline Top-3: {baseline_results['top3_accuracy']:.4f}")

    # ---- Feature engineering ----
    print("\n[3/7] Engineering features...")
    X, feature_names, pipeline = prepare_features(df, fit=True)
    print(f"  -> {len(feature_names)} features: {feature_names[:5]}...")

    # Encode target
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(CROP_LABELS)
    y = le.transform(df["target_crop"].values)
    groups = df["location_id"].values

    # Yield targets
    y_yield = df["yield_kg_ha"].values if "yield_kg_ha" in df.columns else np.zeros(len(df))

    # Save feature pipeline
    save_pipeline(pipeline, str(models_dir / "feature_pipeline.joblib"))

    # ---- Train/test split by location ----
    print("\n[4/7] Splitting data (geographic hold-out by location_id)...")
    unique_locs = np.unique(groups)
    n_test_locs = max(1, len(unique_locs) // 5)  # ~20% locations for test
    np.random.seed(42)
    test_locs = set(np.random.choice(unique_locs, size=n_test_locs, replace=False))
    test_mask = np.isin(groups, list(test_locs))
    train_mask = ~test_mask

    X_train, X_test = X.values[train_mask], X.values[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    groups_train = groups[train_mask]
    y_yield_train, y_yield_test = y_yield[train_mask], y_yield[test_mask]

    print(f"  -> Train: {len(X_train)} samples ({sum(train_mask)} locs)")
    print(f"  -> Test:  {len(X_test)} samples ({n_test_locs} held-out locs)")

    # ---- Start MLflow run ----
    if MLFLOW_AVAILABLE:
        mlflow_path = Path("mlflow").resolve()
        mlflow.set_tracking_uri(mlflow_path.as_uri())
        mlflow.set_experiment("crop_recommendation")
        mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        mlflow.log_params({
            "model_type": model_type,
            "n_samples": len(df),
            "n_features": len(feature_names),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_crops": len(CROP_LABELS),
            "tune": tune,
        })

    # ---- Hyperparameter tuning ----
    if tune:
        print(f"\n[5/7] Tuning hyperparameters ({n_tune_trials} trials)...")
        best_params = tune_hyperparameters(
            X_train, y_train, groups_train,
            model_type=model_type, n_trials=n_tune_trials,
            n_classes=len(CROP_LABELS),
        )
        print(f"  -> Best params: {json.dumps({k: v for k, v in best_params.items() if k not in ('objective', 'num_class', 'n_jobs', 'random_state', 'eval_metric', 'use_label_encoder', 'verbose')}, indent=2)}")
    else:
        print("\n[5/7] Using default hyperparameters (use --tune for optimization)...")
        best_params = None

    # ---- Train final classifier ----
    print(f"\n[6/7] Training {model_type} classifier...")
    start_time = time.time()

    clf = CropClassifier(model_type=model_type, n_classes=len(CROP_LABELS), params=best_params)
    clf.fit(X_train, y_train, feature_names=list(X.columns))
    train_time = time.time() - start_time
    print(f"  -> Training completed in {train_time:.1f}s")

    # Evaluate on test set
    eval_results = clf.evaluate(X_test, y_test)
    print(f"  -> Test Top-1 accuracy: {eval_results['top1_accuracy']:.4f}")
    print(f"  -> Test Top-3 accuracy: {eval_results['top3_accuracy']:.4f}")

    # Save classifier
    clf.save(str(models_dir / "crop_classifier.joblib"))

    # ---- Train yield model ----
    print("\n[7/7] Training yield predictor...")
    yield_model = YieldPredictor()
    yield_model.fit(
        X_train, y_yield_train,
        feature_names=list(X.columns),
        yield_by_crop=df[["target_crop", "yield_kg_ha"]].dropna(),
    )
    yield_eval = yield_model.evaluate(X_test, y_yield_test,
                                       crop_labels=le.inverse_transform(y_test))
    print(f"  -> Yield MAE: {yield_eval['overall_mae']}")
    yield_model.save(str(models_dir / "yield_predictor.joblib"))

    # ---- Log to MLflow ----
    if MLFLOW_AVAILABLE:
        mlflow.log_metrics({
            "baseline_top1_accuracy": baseline_results["top1_accuracy"],
            "baseline_top3_accuracy": baseline_results["top3_accuracy"],
            "test_top1_accuracy": eval_results["top1_accuracy"],
            "test_top3_accuracy": eval_results["top3_accuracy"],
            "yield_mae": yield_eval["overall_mae"],
            "train_time_seconds": train_time,
        })
        mlflow.log_artifact(str(models_dir / "crop_classifier.joblib"))
        mlflow.log_artifact(str(models_dir / "yield_predictor.joblib"))
        mlflow.log_artifact(str(models_dir / "feature_pipeline.joblib"))
        mlflow.end_run()
        print("\n  -> MLflow run logged successfully")

    # ---- Summary ----
    results = {
        "model_type": model_type,
        "baseline": baseline_results,
        "test_metrics": eval_results,
        "yield_metrics": yield_eval,
        "train_time_seconds": train_time,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    results_path = models_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Success metric check
    target = 0.75
    if eval_results["top3_accuracy"] >= target:
        print(f"\n[OK] SUCCESS: Top-3 accuracy {eval_results['top3_accuracy']:.4f} >= {target}")
    else:
        print(f"\n[!] WARNING: Top-3 accuracy {eval_results['top3_accuracy']:.4f} < {target}")
        print("  Remediation: increase data, tune hyperparameters, add features")

    print("=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train crop recommendation models")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--model-type", default="xgboost",
                        choices=["xgboost", "lightgbm"], help="Model backend")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--tune-trials", type=int, default=30, help="Number of Optuna trials")
    args = parser.parse_args()
    train(args.data_dir, args.model_type, args.tune, args.tune_trials)
