"""Tests for FastAPI /predict endpoint."""

import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# We need to mock the models before importing app
@pytest.fixture
def mock_models():
    """Create mock model objects."""
    from src.data.schema import CROP_LABELS
    from sklearn.preprocessing import LabelEncoder

    # Mock classifier
    mock_clf = MagicMock()
    mock_clf.feature_names = [
        "N_kg_ha", "P_kg_ha", "K_kg_ha", "pH",
        "avg_temp_c", "humidity_pct", "avg_precip_mm",
        "N_P_ratio", "N_K_ratio", "P_K_ratio",
    ]
    mock_clf.model = MagicMock()
    mock_clf.predict_top_k.return_value = [[
        {"crop": "rice", "probability": 0.65, "class_idx": 0},
        {"crop": "maize", "probability": 0.20, "class_idx": 1},
        {"crop": "chickpea", "probability": 0.10, "class_idx": 2},
    ]]
    mock_clf.label_encoder = LabelEncoder()
    mock_clf.label_encoder.fit(CROP_LABELS)

    # Mock yield model
    mock_yield = MagicMock()
    mock_yield.predict_for_crop.return_value = 2500.0

    # Mock explainer
    mock_explainer = MagicMock()
    mock_explainer.explain_prediction.return_value = [
        {"feature": "N_kg_ha", "contribution": 0.15},
        {"feature": "avg_precip_mm", "contribution": 0.12},
        {"feature": "pH", "contribution": -0.05},
    ]

    return mock_clf, mock_yield, mock_explainer


@pytest.fixture
def client(mock_models):
    """Create test client with mocked models."""
    mock_clf, mock_yield, mock_explainer = mock_models

    from src.data.features import build_feature_pipeline
    mock_pipeline = build_feature_pipeline()

    import src.api.app as app_module
    app_module.classifier = mock_clf
    app_module.yield_model = mock_yield
    app_module.shap_explainer = mock_explainer
    app_module.feature_pipeline = mock_pipeline
    app_module.data_checksum = "test_checksum_1234"
    app_module.model_version = "1.0.0-test"

    from fastapi.testclient import TestClient
    return TestClient(app_module.app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded")
        assert "model_loaded" in data
        assert "version" in data


class TestPredictEndpoint:
    def test_predict_returns_correct_schema(self, client):
        payload = {
            "N": 90, "P": 42, "K": 43,
            "temperature": 20.87, "humidity": 82.0,
            "ph": 6.5, "rainfall": 202.9,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()

        # Validate response schema
        assert "top_3" in data
        assert "model_version" in data
        assert "data_checksum" in data
        assert len(data["top_3"]) == 3

        for pred in data["top_3"]:
            assert "crop" in pred
            assert "probability" in pred
            assert "expected_yield_kg_ha" in pred
            assert "explanation" in pred
            assert isinstance(pred["explanation"], list)
            for exp in pred["explanation"]:
                assert "feature" in exp
                assert "contribution" in exp

    def test_predict_missing_field_returns_422(self, client):
        payload = {"N": 90, "P": 42}  # Missing required fields
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_invalid_range_returns_422(self, client):
        payload = {
            "N": -10, "P": 42, "K": 43,  # N < 0
            "temperature": 20.87, "humidity": 82.0,
            "ph": 6.5, "rainfall": 202.9,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_all_crops_valid(self, client):
        """All recommended crops should be valid crop names."""
        from src.data.schema import CROP_LABELS
        payload = {
            "N": 90, "P": 42, "K": 43,
            "temperature": 20.87, "humidity": 82.0,
            "ph": 6.5, "rainfall": 202.9,
        }
        response = client.post("/predict", json=payload)
        data = response.json()
        for pred in data["top_3"]:
            assert pred["crop"] in CROP_LABELS


class TestUIEndpoint:
    def test_root_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
