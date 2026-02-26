"""
Feature engineering pipeline for crop recommendation.
Saved as a scikit-learn Pipeline artifact for consistent transforms.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from pathlib import Path


class PHBinner(BaseEstimator, TransformerMixin):
    """Bin pH into agronomic categories."""

    PH_BINS = [0, 4.5, 5.5, 6.5, 7.5, 8.5, 14.0]
    PH_LABELS = [
        "strongly_acidic", "acidic", "slightly_acidic",
        "neutral", "alkaline", "strongly_alkaline",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "pH" in X.columns:
            X["pH_bin"] = pd.cut(
                X["pH"], bins=self.PH_BINS, labels=self.PH_LABELS,
                include_lowest=True,
            ).astype(str)
        return X


class NutrientRatios(BaseEstimator, TransformerMixin):
    """Compute nutrient ratios: N/P, N/K, P/K."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        eps = 1e-6
        if "N_kg_ha" in X.columns and "P_kg_ha" in X.columns:
            X["N_P_ratio"] = X["N_kg_ha"] / (X["P_kg_ha"] + eps)
        if "N_kg_ha" in X.columns and "K_kg_ha" in X.columns:
            X["N_K_ratio"] = X["N_kg_ha"] / (X["K_kg_ha"] + eps)
        if "P_kg_ha" in X.columns and "K_kg_ha" in X.columns:
            X["P_K_ratio"] = X["P_kg_ha"] / (X["K_kg_ha"] + eps)
        return X


class RainfallBinner(BaseEstimator, TransformerMixin):
    """Bin rainfall into categories."""

    RAIN_BINS = [0, 50, 100, 200, 500, 5000]
    RAIN_LABELS = ["very_low", "low", "medium", "high", "very_high"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "avg_precip_mm" in X.columns:
            X["rainfall_bin"] = pd.cut(
                X["avg_precip_mm"], bins=self.RAIN_BINS, labels=self.RAIN_LABELS,
                include_lowest=True,
            ).astype(str)
        return X


class TemperatureBinner(BaseEstimator, TransformerMixin):
    """Bin temperature into zones."""

    TEMP_BINS = [-10, 10, 20, 30, 40, 60]
    TEMP_LABELS = ["cold", "cool", "moderate", "warm", "hot"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "avg_temp_c" in X.columns:
            X["temp_bin"] = pd.cut(
                X["avg_temp_c"], bins=self.TEMP_BINS, labels=self.TEMP_LABELS,
                include_lowest=True,
            ).astype(str)
        return X


class HumidityBinner(BaseEstimator, TransformerMixin):
    """Bin humidity into categories."""

    HUM_BINS = [0, 30, 60, 80, 100]
    HUM_LABELS = ["low", "moderate", "high", "very_high"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "humidity_pct" in X.columns:
            X["humidity_bin"] = pd.cut(
                X["humidity_pct"], bins=self.HUM_BINS, labels=self.HUM_LABELS,
                include_lowest=True,
            ).astype(str)
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode categorical bin columns for model input."""

    BIN_COLS = ["pH_bin", "rainfall_bin", "temp_bin", "humidity_bin"]

    def __init__(self):
        self.dummies_columns_ = []

    def fit(self, X, y=None):
        self.dummies_columns_ = []
        X_t = X.copy()
        for col in self.BIN_COLS:
            if col in X_t.columns:
                dummies = pd.get_dummies(X_t[col], prefix=col, drop_first=False)
                self.dummies_columns_.extend(dummies.columns.tolist())
        self.dummies_columns_ = sorted(set(self.dummies_columns_))
        return self

    def transform(self, X):
        X_t = X.copy()
        for col in self.BIN_COLS:
            if col in X_t.columns:
                dummies = pd.get_dummies(X_t[col], prefix=col, drop_first=False)
                X_t = pd.concat([X_t, dummies], axis=1)
                X_t.drop(columns=[col], inplace=True)

        # Ensure all expected dummy columns exist
        for c in self.dummies_columns_:
            if c not in X_t.columns:
                X_t[c] = 0

        return X_t


# ---------- Model-ready feature columns ----------
NUMERIC_FEATURES = [
    "N_kg_ha", "P_kg_ha", "K_kg_ha", "pH",
    "avg_temp_c", "humidity_pct", "avg_precip_mm",
    "N_P_ratio", "N_K_ratio", "P_K_ratio",
]


def build_feature_pipeline() -> Pipeline:
    """Build the full feature engineering pipeline."""
    return Pipeline([
        ("ph_binner", PHBinner()),
        ("nutrient_ratios", NutrientRatios()),
        ("rainfall_binner", RainfallBinner()),
        ("temp_binner", TemperatureBinner()),
        ("humidity_binner", HumidityBinner()),
        ("categorical_encoder", CategoricalEncoder()),
    ])


def get_model_features(df: pd.DataFrame) -> list:
    """Get the list of feature columns used for model training."""
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    # Add one-hot encoded columns
    cat_cols = [c for c in df.columns if any(
        c.startswith(p) for p in ["pH_bin_", "rainfall_bin_", "temp_bin_", "humidity_bin_"]
    )]
    return sorted(numeric + cat_cols)


def prepare_features(df: pd.DataFrame, pipeline: Pipeline = None,
                     fit: bool = True) -> tuple:
    """
    Apply feature pipeline and return (X_features_df, feature_names).
    """
    if pipeline is None:
        pipeline = build_feature_pipeline()

    if fit:
        df_transformed = pipeline.fit_transform(df)
    else:
        df_transformed = pipeline.transform(df)

    feature_cols = get_model_features(df_transformed)
    X = df_transformed[feature_cols].astype(float).fillna(0)
    return X, feature_cols, pipeline


def save_pipeline(pipeline: Pipeline, path: str):
    """Save the feature pipeline artifact."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Feature pipeline saved to {path}")


def load_pipeline(path: str) -> Pipeline:
    """Load a saved feature pipeline."""
    return joblib.load(path)
