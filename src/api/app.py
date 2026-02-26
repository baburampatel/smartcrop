"""
FastAPI application for crop recommendation inference.

Endpoints:
    POST /predict              — Returns top-3 crop recommendations with SHAP explanations
    POST /predict-by-location  — Same, but auto-fetches soil/weather from a PIN or lat,lon
    GET  /health               — Health check
    GET  /                     — Serves field officer UI
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from prometheus_client import Counter, Histogram, generate_latest
    from fastapi.responses import Response as PrometheusResponse
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.schemas import (
    PredictionRequest, PredictionResponse, CropPrediction,
    FeatureContribution, HealthResponse, ExportSuggestion,
    LocationPredictionRequest, LocationPredictionResponse, LocationDataSources,
)
from src.location.pipeline import LocationPipeline, PipelineConfig
from src.data.schema import KAGGLE_CROP_REC_COLUMN_MAP
from src.data.features import load_pipeline, get_model_features
from src.models.classifier import CropClassifier
from src.models.yield_model import YieldPredictor
from src.explain.shap_explain import SHAPExplainer

# ---- App setup ----
app = FastAPI(
    title="Crop Recommendation API",
    description="Production-grade crop recommendation service for Indian agriculture",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Prometheus metrics ----
if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter("predict_requests_total", "Total prediction requests")
    REQUEST_LATENCY = Histogram(
        "predict_latency_seconds", "Prediction latency",
        buckets=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
    )
    PREDICTION_DRIFT = Counter(
        "prediction_crop_total", "Predictions by crop",
        ["crop"],
    )

# ---- Global model references ----
classifier: CropClassifier = None
yield_model: YieldPredictor = None
feature_pipeline = None
shap_explainer: SHAPExplainer = None
data_checksum: str = "unknown"
model_version: str = "1.0.0"


def load_models():
    """Load all model artifacts on startup."""
    global classifier, yield_model, feature_pipeline, shap_explainer, data_checksum

    models_dir = PROJECT_ROOT / "models"

    # Load classifier
    clf_path = models_dir / "crop_classifier.joblib"
    if clf_path.exists():
        classifier = CropClassifier.load(str(clf_path))
        print(f"[OK] Classifier loaded from {clf_path}")
    else:
        print(f"[!!] Classifier not found at {clf_path}")

    # Load yield model
    yield_path = models_dir / "yield_predictor.joblib"
    if yield_path.exists():
        yield_model = YieldPredictor.load(str(yield_path))
        print(f"[OK] Yield model loaded from {yield_path}")
    else:
        print(f"[!!] Yield model not found at {yield_path}")

    # Load feature pipeline
    pipeline_path = models_dir / "feature_pipeline.joblib"
    if pipeline_path.exists():
        feature_pipeline = load_pipeline(str(pipeline_path))
        print(f"[OK] Feature pipeline loaded from {pipeline_path}")
    else:
        print(f"[!!] Feature pipeline not found at {pipeline_path}")

    # Initialize SHAP explainer
    if classifier is not None:
        shap_explainer = SHAPExplainer(classifier.model, classifier.feature_names)
        print("[OK] SHAP explainer initialized")

    # Load data checksum
    metadata_path = PROJECT_ROOT / "data" / "processed" / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
            data_checksum = meta.get("output_checksum_sha256", "unknown")[:16]


@app.on_event("startup")
async def startup_event():
    load_models()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if classifier is not None else "degraded",
        model_loaded=classifier is not None,
        version=model_version,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Generate top-3 crop recommendations with expected yield and SHAP explanations.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    # Convert request to DataFrame with canonical column names
    input_data = {
        "N_kg_ha": request.N,
        "P_kg_ha": request.P,
        "K_kg_ha": request.K,
        "avg_temp_c": request.temperature,
        "humidity_pct": request.humidity,
        "pH": request.ph,
        "avg_precip_mm": request.rainfall,
    }
    df_input = pd.DataFrame([input_data])

    # Apply feature pipeline
    if feature_pipeline is not None:
        df_transformed = feature_pipeline.transform(df_input)
    else:
        df_transformed = df_input

    feature_cols = get_model_features(df_transformed)
    # Ensure all expected features are present
    for col in classifier.feature_names:
        if col not in df_transformed.columns:
            df_transformed[col] = 0

    X = df_transformed[classifier.feature_names].astype(float).fillna(0).values

    # Get top-3 predictions
    top_k = classifier.predict_top_k(X, k=3)

    # Build response
    results = []
    for pred in top_k[0]:
        crop = pred["crop"]
        probability = pred["probability"]

        # Get yield prediction
        expected_yield = 0.0
        if yield_model is not None:
            expected_yield = yield_model.predict_for_crop(X, crop)

        # Get SHAP explanation
        explanation = []
        if shap_explainer is not None:
            explanation = shap_explainer.explain_prediction(
                X, pred["class_idx"], top_features=5
            )

        results.append(CropPrediction(
            crop=crop,
            probability=probability,
            expected_yield_kg_ha=round(expected_yield, 1),
            explanation=[FeatureContribution(**e) for e in explanation],
        ))

    latency = time.time() - start_time

    # Prometheus metrics
    if PROMETHEUS_AVAILABLE:
        REQUEST_COUNT.inc()
        REQUEST_LATENCY.observe(latency)
        for r in results:
            PREDICTION_DRIFT.labels(crop=r.crop).inc()

    return PredictionResponse(
        top_3=results,
        model_version=model_version,
        data_checksum=data_checksum,
    )


# ---- Prometheus metrics endpoint ----
if PROMETHEUS_AVAILABLE:
    @app.get("/metrics")
    async def metrics():
        return PrometheusResponse(
            content=generate_latest(),
            media_type="text/plain",
        )


# ---- Location-based prediction endpoint ----
@app.post("/predict-by-location", response_model=LocationPredictionResponse)
async def predict_by_location(request: LocationPredictionRequest):
    """
    Auto-fetch soil and weather data for a location, then generate
    top-3 crop recommendations with expected yield and SHAP explanations.

    Accepts an Indian PIN code or lat,lon coordinates.
    """
    # Run the location pipeline to acquire data
    config = PipelineConfig(
        location=request.location,
        pipeline=request.pipeline,
        target_crops=request.target_crops,
        prioritize=request.prioritize,
    )
    loc_pipeline = LocationPipeline()

    try:
        pipe_result = loc_pipeline.run(config)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Data acquisition failed: {e}",
        )

    mi = pipe_result.model_input

    # If model is loaded, run prediction
    if classifier is not None:
        start_time = time.time()
        input_data = {
            "N_kg_ha": mi["N"],
            "P_kg_ha": mi["P"],
            "K_kg_ha": mi["K"],
            "avg_temp_c": mi["temperature"],
            "humidity_pct": mi["humidity"],
            "pH": mi["ph"],
            "avg_precip_mm": mi["rainfall"],
        }
        df_input = pd.DataFrame([input_data])

        if feature_pipeline is not None:
            df_transformed = feature_pipeline.transform(df_input)
        else:
            df_transformed = df_input

        for col in classifier.feature_names:
            if col not in df_transformed.columns:
                df_transformed[col] = 0

        X = df_transformed[classifier.feature_names].astype(float).fillna(0).values
        top_k = classifier.predict_top_k(X, k=3)

        results = []
        for pred in top_k[0]:
            crop = pred["crop"]
            probability = pred["probability"]

            expected_yield = 0.0
            if yield_model is not None:
                expected_yield = yield_model.predict_for_crop(X, crop)

            explanation = []
            if shap_explainer is not None:
                explanation = shap_explainer.explain_prediction(
                    X, pred["class_idx"], top_features=5
                )

            results.append(CropPrediction(
                crop=crop,
                probability=probability,
                expected_yield_kg_ha=round(expected_yield, 1),
                explanation=[FeatureContribution(**e) for e in explanation],
            ))
    else:
        # Model not loaded — return empty predictions with data only
        results = []
        pipe_result.warnings.append("Model not loaded; returning data only")

    return LocationPredictionResponse(
        top_3=results,
        model_version=model_version,
        data_checksum=data_checksum,
        location_info=pipe_result.location_info,
        soil_data=pipe_result.soil_data,
        weather_data=pipe_result.weather_data,
        market_data=pipe_result.market_data,
        data_sources=LocationDataSources(
            sources=pipe_result.data_sources,
            warnings=pipe_result.warnings,
            pipeline_used=pipe_result.pipeline_used,
        ),
        regional_crops=pipe_result.regional_crops,
        satellite_data=pipe_result.satellite_data,
        export_suggestion=ExportSuggestion(**pipe_result.export_suggestion) if pipe_result.export_suggestion else None,
    )


# ---- Serve UI ----
UI_DIR = PROJECT_ROOT / "ui"


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the field officer web UI."""
    ui_path = UI_DIR / "index.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Crop Recommendation API</h1><p>Visit /docs for API documentation.</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
