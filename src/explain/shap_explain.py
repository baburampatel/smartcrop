"""
SHAP-based explainability for crop recommendation predictions.
Generates global feature importance and per-prediction explanations.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:
    shap = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class SHAPExplainer:
    """
    SHAP explanation generator for tree-based crop recommendation models.
    """

    def __init__(self, model, feature_names: List[str]):
        """
        model: fitted XGBoost or LightGBM model object (.model attribute of CropClassifier)
        feature_names: list of feature column names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        """Initialize SHAP TreeExplainer."""
        if shap is None:
            print("WARNING: SHAP not installed. Explanations will use feature importance fallback.")
            return
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            print(f"WARNING: Could not create TreeExplainer: {e}. Using fallback.")

    def explain_prediction(self, X: np.ndarray, predicted_class: int,
                           top_features: int = 5) -> List[Dict]:
        """
        Generate per-prediction SHAP explanation for a single sample.
        Returns list of {feature, contribution} dicts for the predicted class.
        """
        if self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(X[:1])
                # For multi-class, shap_values is a list of arrays (one per class)
                if isinstance(shap_values, list):
                    class_shap = shap_values[predicted_class][0]
                elif len(shap_values.shape) == 3:
                    class_shap = shap_values[0, :, predicted_class]
                else:
                    class_shap = shap_values[0]

                # Get top contributing features
                indices = np.argsort(np.abs(class_shap))[::-1][:top_features]
                explanation = []
                for idx in indices:
                    fname = self.feature_names[idx] if idx < len(self.feature_names) else f"f{idx}"
                    explanation.append({
                        "feature": fname,
                        "contribution": float(round(class_shap[idx], 4)),
                    })
                return explanation
            except Exception as e:
                pass

        # Fallback: use model feature importance
        return self._importance_fallback(top_features)

    def _importance_fallback(self, top_features: int = 5) -> List[Dict]:
        """Fallback to feature importance if SHAP fails."""
        try:
            if hasattr(self.model, "feature_importances_"):
                importance = self.model.feature_importances_
                indices = np.argsort(importance)[::-1][:top_features]
                return [
                    {
                        "feature": self.feature_names[i] if i < len(self.feature_names) else f"f{i}",
                        "contribution": float(round(importance[i], 4)),
                    }
                    for i in indices
                ]
        except Exception:
            pass
        return [{"feature": "unknown", "contribution": 0.0}]

    def global_feature_importance(self, X: np.ndarray,
                                   max_samples: int = 200) -> Dict[str, float]:
        """
        Compute global feature importance using mean |SHAP values|.
        """
        if self.explainer is not None:
            try:
                X_sample = X[:min(len(X), max_samples)]
                shap_values = self.explainer.shap_values(X_sample)

                # Average across all classes and samples
                if isinstance(shap_values, list):
                    all_shap = np.abs(np.array(shap_values)).mean(axis=(0, 1))
                elif len(shap_values.shape) == 3:
                    all_shap = np.abs(shap_values).mean(axis=(0, 2))
                else:
                    all_shap = np.abs(shap_values).mean(axis=0)

                importance = {}
                for i, val in enumerate(all_shap):
                    fname = self.feature_names[i] if i < len(self.feature_names) else f"f{i}"
                    importance[fname] = float(round(val, 4))

                return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            except Exception as e:
                print(f"SHAP global importance failed: {e}")

        # Fallback
        return {f: 0.0 for f in self.feature_names}
