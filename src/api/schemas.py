"""
Pydantic request/response schemas for the FastAPI crop recommendation service.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input schema for /predict endpoint."""
    N: float = Field(..., ge=0, le=500, description="Nitrogen content (kg/ha)")
    P: float = Field(..., ge=0, le=500, description="Phosphorous content (kg/ha)")
    K: float = Field(..., ge=0, le=500, description="Potassium content (kg/ha)")
    temperature: float = Field(..., ge=-10, le=60, description="Temperature (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    ph: float = Field(..., ge=0, le=14, description="pH value of soil")
    rainfall: float = Field(..., ge=0, le=5000, description="Rainfall (mm)")

    # Optional extended features
    EC_dS_m: Optional[float] = Field(None, description="Electrical conductivity (dS/m)")
    OC_pct: Optional[float] = Field(None, description="Organic carbon (%)")
    S_mg_kg: Optional[float] = Field(None, description="Sulphur (mg/kg)")
    zn_ppm: Optional[float] = Field(None, description="Zinc (ppm)")
    fe_ppm: Optional[float] = Field(None, description="Iron (ppm)")
    mn_ppm: Optional[float] = Field(None, description="Manganese (ppm)")
    cu_ppm: Optional[float] = Field(None, description="Copper (ppm)")
    season: Optional[str] = Field(None, description="Season (kharif/rabi)")
    previous_crop: Optional[str] = Field(None, description="Previous crop grown")
    irrigation: Optional[str] = Field(None, description="Irrigation type")

    model_config = {"json_schema_extra": {
        "examples": [{
            "N": 90, "P": 42, "K": 43, "temperature": 20.87,
            "humidity": 82.0, "ph": 6.5, "rainfall": 202.9,
        }]
    }}


class FeatureContribution(BaseModel):
    """SHAP feature contribution."""
    feature: str
    contribution: float


class CropPrediction(BaseModel):
    """Single crop prediction result."""
    crop: str
    probability: float
    expected_yield_kg_ha: float
    explanation: List[FeatureContribution]



class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str


class LocationPredictionRequest(BaseModel):
    """Input schema for /predict-by-location endpoint."""
    location: str = Field(
        ...,
        description="Indian PIN code (e.g., '560001') or lat,lon (e.g., '12.97,77.59')",
    )
    target_crops: Optional[List[str]] = Field(
        None,
        description="Crops to consider (default: rice,maize,wheat,soybean,groundnut,sugarcane)",
    )
    pipeline: str = Field(
        "Tertiary",
        description="Pipeline tier: Primary, Secondary, or Tertiary",
    )
    prioritize: str = Field(
        "data_quality",
        description="Optimization priority: data_quality, cost, or speed",
    )
    season: Optional[str] = Field(
        None,
        description="Cropping season: kharif, rabi, or zaid",
    )

    model_config = {"json_schema_extra": {
        "examples": [{
            "location": "560001",
            "pipeline": "Tertiary",
        }]
    }}


class ExportSuggestion(BaseModel):
    """Export opportunity suggestion for a single crop."""
    crop: str
    export_score: float
    reason: str
    reason_bullets: List[str] = []
    expected_export_window: str = "next_quarter"
    nearest_port: str = ""
    port_distance_km: float = 0
    confidence: float = 0
    avg_price_usd_ton: float = 0
    cold_chain_required: bool = False
    processing_required: str = ""


class LocationDataSources(BaseModel):
    """Metadata about data sources used in the prediction."""
    sources: List[str]
    warnings: List[str]
    pipeline_used: str


class PredictionResponse(BaseModel):
    """Output schema for /predict endpoint."""
    top_3: List[CropPrediction]
    model_version: str
    data_checksum: str
    export_suggestion: Optional[ExportSuggestion] = None


class LocationPredictionResponse(BaseModel):
    """Output schema for /predict-by-location endpoint."""
    top_3: List[CropPrediction]
    model_version: str
    data_checksum: str
    location_info: Dict
    soil_data: Dict
    weather_data: Dict
    market_data: Dict
    data_sources: LocationDataSources
    regional_crops: Optional[List[Dict]] = None
    satellite_data: Optional[Dict] = None
    export_suggestion: Optional[ExportSuggestion] = None

