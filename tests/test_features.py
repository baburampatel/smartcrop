"""Tests for feature engineering pipeline."""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import (
    PHBinner, NutrientRatios, RainfallBinner, TemperatureBinner,
    HumidityBinner, CategoricalEncoder, build_feature_pipeline,
    prepare_features, NUMERIC_FEATURES,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "N_kg_ha": [90.0, 20.0, 50.0],
        "P_kg_ha": [42.0, 68.0, 30.0],
        "K_kg_ha": [43.0, 80.0, 20.0],
        "pH": [6.5, 7.2, 4.0],
        "avg_temp_c": [25.0, 18.0, 35.0],
        "humidity_pct": [80.0, 15.0, 92.0],
        "avg_precip_mm": [200.0, 80.0, 30.0],
    })


class TestPHBinner:
    def test_bins_correctly(self, sample_df):
        binner = PHBinner()
        result = binner.transform(sample_df)
        assert "pH_bin" in result.columns
        # pH 6.5 → slightly_acidic
        assert result["pH_bin"].iloc[0] == "slightly_acidic"
        # pH 7.2 → neutral
        assert result["pH_bin"].iloc[1] == "neutral"
        # pH 4.0 → strongly_acidic
        assert result["pH_bin"].iloc[2] == "strongly_acidic"


class TestNutrientRatios:
    def test_creates_ratios(self, sample_df):
        ratios = NutrientRatios()
        result = ratios.transform(sample_df)
        assert "N_P_ratio" in result.columns
        assert "N_K_ratio" in result.columns
        assert "P_K_ratio" in result.columns
        # N=90, P=42 → N/P ≈ 2.14
        assert abs(result["N_P_ratio"].iloc[0] - 90.0/42.0) < 0.1


class TestRainfallBinner:
    def test_bins_correctly(self, sample_df):
        binner = RainfallBinner()
        result = binner.transform(sample_df)
        assert "rainfall_bin" in result.columns
        # 200mm → medium
        assert result["rainfall_bin"].iloc[0] == "medium"
        # 30mm → very_low
        assert result["rainfall_bin"].iloc[2] == "very_low"


class TestFullPipeline:
    def test_pipeline_runs(self, sample_df):
        X, feature_cols, pipeline = prepare_features(sample_df, fit=True)
        assert len(feature_cols) > 0
        assert X.shape[0] == 3
        assert X.shape[1] == len(feature_cols)
        # No NaN in output
        assert not X.isna().any().any()

    def test_pipeline_transform_consistency(self, sample_df):
        """Pipeline should produce same output on re-transform."""
        X1, cols1, pipeline = prepare_features(sample_df, fit=True)
        X2, cols2, _ = prepare_features(sample_df, pipeline=pipeline, fit=False)
        assert cols1 == cols2
        np.testing.assert_array_almost_equal(X1.values, X2.values)
