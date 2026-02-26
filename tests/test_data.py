"""Tests for data schema, ingestion, and validation."""

import sys
import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import (
    CANONICAL_COLUMNS, REQUIRED_INFERENCE_FEATURES, CROP_LABELS,
    FEATURE_RANGES, normalize_crop_name, rename_kaggle_columns,
    compute_yield, CROP_TO_IDX, IDX_TO_CROP,
)


class TestSchema:
    """Test canonical schema definitions."""

    def test_crop_labels_count(self):
        assert len(CROP_LABELS) == 22

    def test_crop_labels_sorted(self):
        assert CROP_LABELS == sorted(CROP_LABELS)

    def test_crop_index_mapping(self):
        for crop in CROP_LABELS:
            idx = CROP_TO_IDX[crop]
            assert IDX_TO_CROP[idx] == crop

    def test_required_features_subset(self):
        """Required inference features must be valid column names."""
        assert len(REQUIRED_INFERENCE_FEATURES) == 7

    def test_feature_ranges_valid(self):
        for col, (lo, hi) in FEATURE_RANGES.items():
            assert lo < hi, f"Invalid range for {col}: [{lo}, {hi}]"


class TestNormalization:
    """Test data normalization utilities."""

    def test_normalize_crop_name(self):
        assert normalize_crop_name("Rice") == "rice"
        assert normalize_crop_name("  MAIZE  ") == "maize"
        assert normalize_crop_name("chickpea") == "chickpea"

    def test_rename_kaggle_columns(self):
        df = pd.DataFrame({
            "N": [90], "P": [42], "K": [43],
            "temperature": [25], "humidity": [80],
            "ph": [6.5], "rainfall": [200], "label": ["rice"],
        })
        df_renamed = rename_kaggle_columns(df)
        assert "N_kg_ha" in df_renamed.columns
        assert "P_kg_ha" in df_renamed.columns
        assert "avg_temp_c" in df_renamed.columns
        assert "target_crop" in df_renamed.columns
        assert df_renamed["target_crop"].iloc[0] == "rice"

    def test_compute_yield(self):
        df = pd.DataFrame({
            "production_tonnes": [1000.0],
            "area_ha": [500.0],
        })
        df = compute_yield(df)
        assert "yield_kg_ha" in df.columns
        assert df["yield_kg_ha"].iloc[0] == 2000.0

    def test_compute_yield_zero_area(self):
        df = pd.DataFrame({
            "production_tonnes": [100.0],
            "area_ha": [0.0],
        })
        df = compute_yield(df)
        assert np.isnan(df["yield_kg_ha"].iloc[0])
