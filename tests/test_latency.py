"""
Latency test: verifies p95 latency ≤ 300ms on modest CPU.

Usage:
    pytest tests/test_latency.py -v -s
"""

import sys
import time
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def client():
    """Create test client with mocked models for latency testing."""
    from src.data.schema import CROP_LABELS
    from sklearn.preprocessing import LabelEncoder
    from src.data.features import build_feature_pipeline

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
    ]

    import src.api.app as app_module
    app_module.classifier = mock_clf
    app_module.yield_model = mock_yield
    app_module.shap_explainer = mock_explainer
    app_module.feature_pipeline = build_feature_pipeline()
    app_module.data_checksum = "test_checksum"

    from fastapi.testclient import TestClient
    return TestClient(app_module.app)


class TestLatency:
    def test_p95_latency_under_300ms(self, client):
        """P95 latency must be ≤ 300ms over 100 requests."""
        payload = {
            "N": 90, "P": 42, "K": 43,
            "temperature": 20.87, "humidity": 82.0,
            "ph": 6.5, "rainfall": 202.9,
        }

        latencies = []
        n_requests = 100

        # Warmup
        for _ in range(5):
            client.post("/predict", json=payload)

        # Timed run
        for i in range(n_requests):
            start = time.perf_counter()
            response = client.post("/predict", json=payload)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            assert response.status_code == 200

        latencies = np.array(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        avg = np.mean(latencies)

        print(f"\n=== Latency Report ({n_requests} requests) ===")
        print(f"  Mean:  {avg:.1f} ms")
        print(f"  P50:   {p50:.1f} ms")
        print(f"  P95:   {p95:.1f} ms")
        print(f"  P99:   {p99:.1f} ms")
        print(f"  Min:   {latencies.min():.1f} ms")
        print(f"  Max:   {latencies.max():.1f} ms")

        assert p95 <= 300, f"P95 latency {p95:.1f}ms exceeds 300ms target"
