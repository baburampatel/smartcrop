"""
Per-crop yield regression model.
Predicts expected yield (kg/ha) given soil and climate features.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib

try:
    import xgboost as xgb
except ImportError:
    xgb = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import CROP_LABELS


class YieldPredictor:
    """
    Multi-output yield predictor: given soil/climate features,
    predict expected yield for each crop.
    Uses a single XGBoost regressor with crop as an extra feature,
    or a dictionary of per-crop regressors.
    """

    def __init__(self, strategy: str = "single"):
        """
        strategy: 'single' (one model with crop label) or 'per_crop' (one model each)
        """
        self.strategy = strategy
        self.models: Dict[str, object] = {}
        self.global_model = None
        self.feature_names: List[str] = []
        self.crop_yields_fallback: Dict[str, float] = {}  # median yields

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: List[str] = None,
            crop_labels: np.ndarray = None,
            yield_by_crop: pd.DataFrame = None):
        """
        Train yield predictor.
        If yield_by_crop is provided as a DataFrame with (target_crop, yield_kg_ha),
        we compute per-crop median yields as a simple fallback predictor.
        """
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        # Compute fallback median yields
        if yield_by_crop is not None and len(yield_by_crop) > 0:
            self.crop_yields_fallback = (
                yield_by_crop.groupby("target_crop")["yield_kg_ha"]
                .median()
                .to_dict()
            )

        # Train a simple global regressor if we have yield target
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() > 10 and xgb is not None:
            self.global_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
            )
            self.global_model.fit(X[valid_mask], y[valid_mask])
            print(f"Yield model trained on {valid_mask.sum()} samples")
        else:
            print("Using fallback median yields (insufficient data for regression)")

        return self

    def predict(self, X: np.ndarray, crop: str = None) -> np.ndarray:
        """Predict yield for the given features."""
        if self.global_model is not None:
            return self.global_model.predict(X)
        elif crop and crop in self.crop_yields_fallback:
            return np.full(X.shape[0], self.crop_yields_fallback[crop])
        else:
            return np.full(X.shape[0], 0.0)

    def predict_for_crop(self, X: np.ndarray, crop: str) -> float:
        """Predict yield for a specific crop (single sample)."""
        if self.global_model is not None:
            pred = self.global_model.predict(X[:1])
            return float(pred[0])
        return self.crop_yields_fallback.get(crop, 0.0)

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 crop_labels: np.ndarray = None) -> Dict:
        """Compute MAE metrics."""
        valid = ~np.isnan(y)
        if valid.sum() == 0:
            return {"overall_mae": float("nan"), "per_crop_mae": {}}

        y_pred = self.predict(X[valid])
        y_true = y[valid]
        overall_mae = float(np.mean(np.abs(y_true - y_pred)))

        per_crop_mae = {}
        if crop_labels is not None:
            for crop in np.unique(crop_labels[valid]):
                mask = crop_labels[valid] == crop
                if mask.sum() > 0:
                    mae = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
                    per_crop_mae[str(crop)] = round(mae, 2)

        return {
            "overall_mae": round(overall_mae, 2),
            "per_crop_mae": per_crop_mae,
            "n_samples": int(valid.sum()),
        }

    def save(self, path: str):
        """Save model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "strategy": self.strategy,
            "global_model": self.global_model,
            "crop_yields_fallback": self.crop_yields_fallback,
            "feature_names": self.feature_names,
        }
        joblib.dump(artifact, path)
        print(f"Yield model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "YieldPredictor":
        artifact = joblib.load(path)
        obj = cls(strategy=artifact["strategy"])
        obj.global_model = artifact["global_model"]
        obj.crop_yields_fallback = artifact["crop_yields_fallback"]
        obj.feature_names = artifact["feature_names"]
        return obj
