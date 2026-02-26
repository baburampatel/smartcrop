"""
XGBoost/LightGBM multi-class classifier for top-3 crop recommendations.
Uses GroupKFold by location_id, Optuna for hyperparameter tuning,
and SMOTE for class imbalance handling.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, top_k_accuracy_score,
    classification_report, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import CROP_LABELS, CROP_TO_IDX, IDX_TO_CROP


class CropClassifier:
    """
    Multi-class crop recommendation classifier.
    Supports XGBoost and LightGBM backends.
    """

    def __init__(self, model_type: str = "xgboost", n_classes: int = 22, params: dict = None):
        self.model_type = model_type
        self.n_classes = n_classes
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(CROP_LABELS)
        self.model = None
        self.params = params or self._default_params()
        self.feature_names: List[str] = []

    def _default_params(self) -> dict:
        if self.model_type == "xgboost":
            return {
                "objective": "multi:softprob",
                "num_class": self.n_classes,
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 200,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42,
                "n_jobs": -1,
                "eval_metric": "mlogloss",
                "use_label_encoder": False,
            }
        elif self.model_type == "lightgbm":
            return {
                "objective": "multiclass",
                "num_class": self.n_classes,
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 200,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_samples": 10,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }
        return {}

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """Train the classifier."""
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        if self.model_type == "xgboost" and xgb is not None:
            params = {k: v for k, v in self.params.items()
                      if k not in ("n_estimators", "use_label_encoder")}
            self.model = xgb.XGBClassifier(
                n_estimators=self.params.get("n_estimators", 200),
                use_label_encoder=False,
                **params,
            )
            self.model.fit(X, y)

        elif self.model_type == "lightgbm" and lgb is not None:
            params = {k: v for k, v in self.params.items() if k != "n_estimators"}
            self.model = lgb.LGBMClassifier(
                n_estimators=self.params.get("n_estimators", 200),
                **params,
            )
            self.model.fit(X, y)
        else:
            raise RuntimeError(f"Model type '{self.model_type}' not available")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class probabilities."""
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions."""
        return self.model.predict(X)

    def predict_top_k(self, X: np.ndarray, k: int = 3) -> List[List[Dict]]:
        """
        Get top-k predictions with probabilities.
        Returns list of lists of {crop, probability} dicts.
        """
        proba = self.predict_proba(X)
        results = []
        for i in range(len(proba)):
            top_indices = np.argsort(proba[i])[::-1][:k]
            preds = []
            for idx in top_indices:
                crop_name = self.label_encoder.inverse_transform([idx])[0]
                preds.append({
                    "crop": crop_name,
                    "probability": float(round(proba[i][idx], 4)),
                    "class_idx": int(idx),
                })
            results.append(preds)
        return results

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Full evaluation metrics."""
        proba = self.predict_proba(X)
        y_pred = self.predict(X)

        top1_acc = accuracy_score(y, y_pred)
        top3_acc = top_k_accuracy_score(y, proba, k=3, labels=range(self.n_classes))

        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y, y_pred).tolist()

        return {
            "top1_accuracy": float(round(top1_acc, 4)),
            "top3_accuracy": float(round(top3_acc, 4)),
            "classification_report": report,
            "confusion_matrix": cm,
        }

    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": self.model,
            "model_type": self.model_type,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "params": self.params,
            "n_classes": self.n_classes,
        }
        joblib.dump(artifact, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "CropClassifier":
        """Load model from disk."""
        artifact = joblib.load(path)
        obj = cls(
            model_type=artifact["model_type"],
            n_classes=artifact["n_classes"],
            params=artifact["params"],
        )
        obj.model = artifact["model"]
        obj.label_encoder = artifact["label_encoder"]
        obj.feature_names = artifact["feature_names"]
        return obj


def tune_hyperparameters(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    model_type: str = "xgboost", n_trials: int = 30,
    n_classes: int = 22,
) -> dict:
    """
    Optuna hyperparameter tuning with GroupKFold cross-validation.
    """
    if optuna is None:
        print("Optuna not available — using default parameters")
        return CropClassifier(model_type)._default_params()

    def objective(trial):
        if model_type == "xgboost":
            params = {
                "objective": "multi:softprob",
                "num_class": n_classes,
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "random_state": 42,
                "n_jobs": -1,
                "eval_metric": "mlogloss",
                "use_label_encoder": False,
            }
        else:
            params = {
                "objective": "multiclass",
                "num_class": n_classes,
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }

        gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        scores = []
        for train_idx, val_idx in gkf.split(X, y, groups):
            clf = CropClassifier(model_type=model_type, n_classes=n_classes, params=params)
            clf.fit(X[train_idx], y[train_idx])
            proba = clf.predict_proba(X[val_idx])
            top3 = top_k_accuracy_score(
                y[val_idx], proba, k=3, labels=range(n_classes)
            )
            scores.append(top3)
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    if model_type == "xgboost":
        best_params.update({
            "objective": "multi:softprob",
            "num_class": n_classes,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "mlogloss",
            "use_label_encoder": False,
        })
    else:
        best_params.update({
            "objective": "multiclass",
            "num_class": n_classes,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        })

    print(f"Best Top-3 accuracy: {study.best_value:.4f}")
    return best_params
