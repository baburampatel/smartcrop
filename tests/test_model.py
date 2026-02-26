"""Tests for model training, prediction, and baseline."""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import CROP_LABELS
from src.models.baseline import predict_rule_based, compute_suitability_score, evaluate_baseline
from src.models.classifier import CropClassifier
from src.models.yield_model import YieldPredictor


class TestBaseline:
    def test_predict_returns_top3(self):
        sample = {
            "N_kg_ha": 90, "P_kg_ha": 42, "K_kg_ha": 43,
            "avg_temp_c": 25, "humidity_pct": 80,
            "pH": 6.5, "avg_precip_mm": 200,
        }
        results = predict_rule_based(sample, top_k=3)
        assert len(results) == 3
        for r in results:
            assert "crop" in r
            assert "probability" in r
            assert "expected_yield_kg_ha" in r
            assert "explanation" in r
            assert r["crop"] in CROP_LABELS or r["crop"] in results[0]["crop"]

    def test_suitability_score_range(self):
        sample = {
            "N_kg_ha": 90, "P_kg_ha": 42, "K_kg_ha": 43,
            "avg_temp_c": 25, "humidity_pct": 80,
            "pH": 6.5, "avg_precip_mm": 200,
        }
        for crop in CROP_LABELS:
            score = compute_suitability_score(sample, crop)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for {crop}"

    def test_probabilities_sum_to_one(self):
        sample = {"N_kg_ha": 50, "P_kg_ha": 50, "K_kg_ha": 50,
                   "avg_temp_c": 25, "humidity_pct": 60, "pH": 7.0,
                   "avg_precip_mm": 150}
        results = predict_rule_based(sample, top_k=3)
        total = sum(r["probability"] for r in results)
        assert abs(total - 1.0) < 0.01


class TestClassifier:
    @pytest.fixture
    def trained_classifier(self):
        """Train a small classifier for testing."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 7)
        y = np.random.randint(0, 5, n)  # 5 classes for speed
        clf = CropClassifier(model_type="xgboost", n_classes=5)
        clf.fit(X, y, feature_names=[f"f{i}" for i in range(7)])
        return clf

    def test_predict_shape(self, trained_classifier):
        X_test = np.random.randn(10, 7)
        preds = trained_classifier.predict(X_test)
        assert preds.shape == (10,)

    def test_predict_proba_shape(self, trained_classifier):
        X_test = np.random.randn(10, 7)
        proba = trained_classifier.predict_proba(X_test)
        assert proba.shape == (10, 5)
        # Each row sums to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_top_k(self, trained_classifier):
        X_test = np.random.randn(3, 7)
        top_k = trained_classifier.predict_top_k(X_test, k=3)
        assert len(top_k) == 3
        for preds in top_k:
            assert len(preds) == 3
            assert all("crop" in p and "probability" in p for p in preds)

    def test_save_load(self, trained_classifier, tmp_path):
        path = str(tmp_path / "test_model.joblib")
        trained_classifier.save(path)
        loaded = CropClassifier.load(path)
        X_test = np.random.randn(5, 7)
        np.testing.assert_array_equal(
            trained_classifier.predict(X_test),
            loaded.predict(X_test),
        )


class TestYieldModel:
    def test_fallback_prediction(self):
        model = YieldPredictor()
        model.crop_yields_fallback = {"rice": 2500.0}
        X = np.random.randn(1, 5)
        result = model.predict_for_crop(X, "rice")
        assert result == 2500.0

    def test_evaluate_returns_dict(self):
        model = YieldPredictor()
        model.crop_yields_fallback = {"rice": 2500.0}
        X = np.random.randn(10, 5)
        y = np.full(10, 2500.0)
        result = model.evaluate(X, y)
        assert "overall_mae" in result
        assert "per_crop_mae" in result
