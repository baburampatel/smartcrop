"""
Location pipeline orchestrator: resolves a LOCATION, fetches soil/weather/market
data from appropriate sources, and produces a model-ready prediction input.

Supports three pipeline tiers:
    Primary   — Lab data + SoilGrids + Weather (requires actual lab results)
    Secondary — Soil Health Card + Weather (scraped data)
    Tertiary  — SoilGrids + Weather only (zero-cost, fully automated)
"""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from src.location.resolver import resolve_location, LocationInfo
from src.location.soilgrids import fetch_soilgrids, estimate_npk_from_soil
from src.location.weather import fetch_weather
from src.location.market import fetch_market_prices
from src.location.cropdata import get_regional_crops, fetch_ndvi_indicator
from src.location.export_scorer import get_export_suggestion

logger = logging.getLogger(__name__)

# Default target crops
DEFAULT_TARGET_CROPS = ["rice", "maize", "wheat", "soybean", "groundnut", "sugarcane"]

# Pipeline tier names
PIPELINE_PRIMARY = "Primary"
PIPELINE_SECONDARY = "Secondary"
PIPELINE_TERTIARY = "Tertiary"


@dataclass
class PipelineConfig:
    """Configuration for a location pipeline run."""
    location: str
    pipeline: str = PIPELINE_TERTIARY
    target_crops: List[str] = None
    prioritize: str = "data_quality"
    radius_km: int = 50
    mode: str = "full_run"
    lab_data: Optional[Dict] = None  # For Primary pipeline: pre-existing lab results

    def __post_init__(self):
        if self.target_crops is None:
            self.target_crops = DEFAULT_TARGET_CROPS.copy()


@dataclass
class PipelineResult:
    """Result from a location pipeline run."""
    location_info: Dict
    soil_data: Dict
    weather_data: Dict
    market_data: Dict
    model_input: Dict
    data_sources: List[str]
    pipeline_used: str
    warnings: List[str]
    regional_crops: List[Dict] = None
    satellite_data: Optional[Dict] = None
    export_suggestion: Optional[Dict] = None


class LocationPipeline:
    """
    Orchestrate location-based data acquisition and produce model-ready inputs.

    Usage:
        pipeline = LocationPipeline()
        result = pipeline.run("560001")
        # result.model_input has keys matching PredictionRequest fields
    """

    def run(self, config: PipelineConfig) -> PipelineResult:
        """
        Execute the pipeline for a given configuration.

        Args:
            config: PipelineConfig with location and settings.

        Returns:
            PipelineResult with all acquired data and model-ready input.
        """
        warnings = []
        data_sources = []

        # Step 1: Resolve location
        logger.info("Step 1: Resolving location '%s'", config.location)
        loc_info = resolve_location(config.location)
        data_sources.append("Nominatim/OpenStreetMap (geocoding)")
        logger.info(
            "Resolved: lat=%.4f, lon=%.4f, state=%s, district=%s",
            loc_info.latitude, loc_info.longitude,
            loc_info.state, loc_info.district,
        )

        # Step 2: Fetch soil data
        logger.info("Step 2: Fetching soil data (pipeline=%s)", config.pipeline)
        soil_data = self._fetch_soil(config, loc_info, data_sources, warnings)

        # Step 3: Fetch weather data
        logger.info("Step 3: Fetching weather data")
        weather_data = fetch_weather(loc_info.latitude, loc_info.longitude)
        if weather_data.get("avg_temp_c") is not None:
            data_sources.append("Open-Meteo ERA5 (weather)")
        else:
            warnings.append("Weather data unavailable; using defaults")

        # Step 4: Fetch market data (optional)
        logger.info("Step 4: Fetching market prices")
        market_data = {}
        if loc_info.state:
            market_data = fetch_market_prices(
                loc_info.state, config.target_crops
            )
            if market_data:
                data_sources.append("Agmarknet/data.gov.in (market prices)")

        # Step 5: Fetch regional crop data (what's grown here)
        logger.info("Step 5: Fetching regional crop data")
        regional_crops = get_regional_crops(loc_info.state)
        if regional_crops:
            data_sources.append("ICAR/MoA Crop Statistics (regional crops)")

        # Step 6: Fetch satellite vegetation data
        logger.info("Step 6: Fetching satellite vegetation data")
        satellite_data = fetch_ndvi_indicator(loc_info.latitude, loc_info.longitude)
        if satellite_data:
            data_sources.append(satellite_data.get("source", "NASA POWER (satellite)"))

        # Step 7: Compute export suggestion
        logger.info("Step 7: Computing export scores")
        export_suggestion = None
        try:
            # Combine target crops + regionally grown crops (deduplicated)
            candidate_crops = list(dict.fromkeys(
                config.target_crops + [c["crop"] for c in (regional_crops or [])[:5]]
            ))
            export_suggestion = get_export_suggestion(
                candidate_crops,
                loc_info.latitude, loc_info.longitude,
                state=loc_info.state,
            )
            if export_suggestion:
                data_sources.append("APEDA/MoC Export Statistics (export scoring)")
        except Exception as e:
            logger.warning("Export scoring failed: %s", e)
            warnings.append(f"Export scoring unavailable: {e}")

        # Step 8: Build model input
        logger.info("Step 8: Building model-ready input")
        model_input = self._build_model_input(
            soil_data, weather_data, config, warnings
        )

        return PipelineResult(
            location_info=asdict(loc_info),
            soil_data=soil_data,
            weather_data=weather_data,
            market_data=market_data,
            model_input=model_input,
            data_sources=data_sources,
            pipeline_used=config.pipeline,
            warnings=warnings,
            regional_crops=regional_crops,
            satellite_data=satellite_data,
            export_suggestion=export_suggestion,
        )

    def _fetch_soil(
        self,
        config: PipelineConfig,
        loc_info: LocationInfo,
        data_sources: List[str],
        warnings: List[str],
    ) -> Dict:
        """Fetch soil data based on the pipeline tier."""
        soil = {}

        if config.pipeline == PIPELINE_PRIMARY:
            # Primary: use provided lab data
            if config.lab_data:
                soil = config.lab_data.copy()
                data_sources.append("Lab soil test (user-provided)")
            else:
                warnings.append(
                    "Primary pipeline selected but no lab_data provided; "
                    "falling back to SoilGrids"
                )
                soil = self._fetch_soilgrids_soil(
                    loc_info, data_sources, warnings
                )

        elif config.pipeline == PIPELINE_SECONDARY:
            # Secondary: would use scraped SHC data
            # For now, fall back to SoilGrids (SHC scraping is a separate tool)
            warnings.append(
                "Secondary pipeline (SHC scraping) not yet automated; "
                "using SoilGrids as proxy"
            )
            soil = self._fetch_soilgrids_soil(loc_info, data_sources, warnings)

        else:
            # Tertiary: SoilGrids only
            soil = self._fetch_soilgrids_soil(loc_info, data_sources, warnings)

        return soil

    def _fetch_soilgrids_soil(
        self,
        loc_info: LocationInfo,
        data_sources: List[str],
        warnings: List[str],
    ) -> Dict:
        """Fetch soil from SoilGrids and estimate NPK."""
        raw_soil = fetch_soilgrids(loc_info.latitude, loc_info.longitude)
        data_sources.append("ISRIC SoilGrids v2.0 (modeled soil)")

        # Check if we got valid data
        if raw_soil.get("pH") is None:
            warnings.append(
                "SoilGrids API returned no data (service may be paused); "
                "using default soil values"
            )
            return {
                "pH": 6.5,
                "N_kg_ha": 50.0,
                "P_kg_ha": 25.0,
                "K_kg_ha": 25.0,
                "source": "defaults",
            }

        # Estimate NPK from soil properties
        npk = estimate_npk_from_soil(raw_soil)

        return {
            "pH": raw_soil.get("pH", 6.5),
            "N_kg_ha": npk["N_kg_ha"],
            "P_kg_ha": npk["P_kg_ha"],
            "K_kg_ha": npk["K_kg_ha"],
            "clay_pct": raw_soil.get("clay_pct"),
            "sand_pct": raw_soil.get("sand_pct"),
            "silt_pct": raw_soil.get("silt_pct"),
            "soc_g_kg": raw_soil.get("soc_g_kg"),
            "source": "soilgrids",
        }

    def _build_model_input(
        self,
        soil: Dict,
        weather: Dict,
        config: PipelineConfig,
        warnings: List[str],
    ) -> Dict:
        """
        Build a model-ready input dict matching PredictionRequest fields.

        Uses soil data + weather data, with sensible defaults for missing values.
        """
        # Defaults for Indian agriculture averages
        defaults = {
            "N": 50.0,
            "P": 25.0,
            "K": 25.0,
            "temperature": 27.0,
            "humidity": 70.0,
            "ph": 6.5,
            "rainfall": 1000.0,
        }

        model_input = {
            "N": soil.get("N_kg_ha", defaults["N"]),
            "P": soil.get("P_kg_ha", defaults["P"]),
            "K": soil.get("K_kg_ha", defaults["K"]),
            "temperature": weather.get("avg_temp_c", defaults["temperature"]),
            "humidity": weather.get("humidity_pct", defaults["humidity"]),
            "ph": soil.get("pH", defaults["ph"]),
            "rainfall": weather.get("avg_precip_mm", defaults["rainfall"]),
        }

        # Track which fields used defaults
        used_defaults = []
        if model_input["N"] == defaults["N"] and soil.get("source") == "defaults":
            used_defaults.append("N")
        if model_input["temperature"] == defaults["temperature"] and weather.get("avg_temp_c") is None:
            used_defaults.append("temperature")
        if model_input["humidity"] == defaults["humidity"] and weather.get("humidity_pct") is None:
            used_defaults.append("humidity")
        if model_input["rainfall"] == defaults["rainfall"] and weather.get("avg_precip_mm") is None:
            used_defaults.append("rainfall")

        if used_defaults:
            warnings.append(f"Used default values for: {', '.join(used_defaults)}")

        return model_input
