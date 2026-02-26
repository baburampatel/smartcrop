"""
API integration tests for the /predict-by-location endpoint.
All HTTP calls and models are mocked.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def mock_pipeline_result():
    """Create a mock LocationPipeline result."""
    from src.location.pipeline import PipelineResult

    return PipelineResult(
        location_info={
            "latitude": 12.9716,
            "longitude": 77.5946,
            "state": "Karnataka",
            "district": "Bangalore Urban",
            "pin_code": None,
            "display_name": "Bengaluru, Karnataka",
            "raw_input": "12.9716,77.5946",
        },
        soil_data={
            "pH": 6.5,
            "N_kg_ha": 80.0,
            "P_kg_ha": 35.0,
            "K_kg_ha": 45.0,
            "source": "soilgrids",
        },
        weather_data={
            "avg_temp_c": 28.0,
            "humidity_pct": 72.0,
            "avg_precip_mm": 1100.0,
            "data_points": 1095,
        },
        market_data={},
        model_input={
            "N": 80.0,
            "P": 35.0,
            "K": 45.0,
            "temperature": 28.0,
            "humidity": 72.0,
            "ph": 6.5,
            "rainfall": 1100.0,
        },
        data_sources=[
            "Nominatim/OpenStreetMap (geocoding)",
            "ISRIC SoilGrids v2.0 (modeled soil)",
            "Open-Meteo ERA5 (weather)",
        ],
        pipeline_used="Tertiary",
        warnings=[],
    )


@pytest.fixture
def mock_models():
    """Create mock model objects (same as test_api.py)."""
    from src.data.schema import CROP_LABELS
    from sklearn.preprocessing import LabelEncoder

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

    mock_yield = MagicMock()
    mock_yield.predict_for_crop.return_value = 2500.0

    mock_explainer = MagicMock()
    mock_explainer.explain_prediction.return_value = [
        {"feature": "N_kg_ha", "contribution": 0.15},
        {"feature": "avg_precip_mm", "contribution": 0.12},
    ]

    return mock_clf, mock_yield, mock_explainer


@pytest.fixture
def client(mock_models, mock_pipeline_result):
    """Create test client with mocked models and pipeline."""
    mock_clf, mock_yield, mock_explainer = mock_models

    from src.data.features import build_feature_pipeline
    mock_pipeline_obj = build_feature_pipeline()

    import src.api.app as app_module
    app_module.classifier = mock_clf
    app_module.yield_model = mock_yield
    app_module.shap_explainer = mock_explainer
    app_module.feature_pipeline = mock_pipeline_obj
    app_module.data_checksum = "test_checksum_1234"
    app_module.model_version = "1.0.0-test"

    from fastapi.testclient import TestClient
    return TestClient(app_module.app), mock_pipeline_result


class TestPredictByLocationEndpoint:

    def test_predict_by_location_returns_correct_schema(self, client):
        """POST /predict-by-location with coords returns full response."""
        test_client, mock_result = client

        with patch("src.api.app.LocationPipeline") as MockPipelineClass:
            mock_instance = MagicMock()
            mock_instance.run.return_value = mock_result
            MockPipelineClass.return_value = mock_instance

            response = test_client.post(
                "/predict-by-location",
                json={"location": "12.9716,77.5946"},
            )

        assert response.status_code == 200
        data = response.json()

        # Validate response fields
        assert "top_3" in data
        assert "model_version" in data
        assert "data_checksum" in data
        assert "location_info" in data
        assert "soil_data" in data
        assert "weather_data" in data
        assert "market_data" in data
        assert "data_sources" in data

        # Validate location info
        assert data["location_info"]["latitude"] == 12.9716
        assert data["location_info"]["state"] == "Karnataka"

        # Validate data sources metadata
        assert data["data_sources"]["pipeline_used"] == "Tertiary"
        assert isinstance(data["data_sources"]["sources"], list)

        # Validate predictions (model is mocked)
        assert len(data["top_3"]) == 3
        for pred in data["top_3"]:
            assert "crop" in pred
            assert "probability" in pred
            assert "expected_yield_kg_ha" in pred
            assert "explanation" in pred

    def test_predict_by_location_with_pin_code(self, client):
        """POST with PIN code format works."""
        test_client, mock_result = client

        with patch("src.api.app.LocationPipeline") as MockPipelineClass:
            mock_instance = MagicMock()
            mock_instance.run.return_value = mock_result
            MockPipelineClass.return_value = mock_instance

            response = test_client.post(
                "/predict-by-location",
                json={"location": "560001", "pipeline": "Tertiary"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "top_3" in data

    def test_predict_by_location_invalid_raises_422(self, client):
        """Invalid location string raises 422."""
        test_client, _ = client

        with patch("src.api.app.LocationPipeline") as MockPipelineClass:
            mock_instance = MagicMock()
            mock_instance.run.side_effect = ValueError("bad location")
            MockPipelineClass.return_value = mock_instance

            response = test_client.post(
                "/predict-by-location",
                json={"location": "not-a-location"},
            )

        assert response.status_code == 422
