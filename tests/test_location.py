"""
Unit tests for the location data fetching modules.
All HTTP calls are mocked — no network access required.
"""

import sys
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------- Resolver tests ----------

class TestResolver:
    def test_resolve_coordinates(self):
        """Parse lat,lon string directly."""
        from src.location.resolver import resolve_location

        mock_reverse = {
            "address": {"state": "Karnataka", "state_district": "Bangalore Urban"},
            "display_name": "Bengaluru, Karnataka, India",
        }
        with patch("src.location.resolver.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_reverse
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            result = resolve_location("12.9716,77.5946")

        assert abs(result.latitude - 12.9716) < 0.001
        assert abs(result.longitude - 77.5946) < 0.001
        assert result.state == "Karnataka"
        assert result.district == "Bangalore Urban"

    def test_resolve_pin_code(self):
        """Geocode an Indian PIN code via Nominatim mock."""
        from src.location.resolver import resolve_location

        mock_response = [{
            "lat": "12.9716",
            "lon": "77.5946",
            "display_name": "560001, Bengaluru, Karnataka, India",
            "address": {
                "state": "Karnataka",
                "state_district": "Bangalore Urban",
            },
        }]
        with patch("src.location.resolver.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_response
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            result = resolve_location("560001")

        assert abs(result.latitude - 12.9716) < 0.001
        assert result.pin_code == "560001"
        assert result.state == "Karnataka"

    def test_resolve_invalid_raises(self):
        """Unrecognized format raises ValueError."""
        from src.location.resolver import resolve_location

        with pytest.raises(ValueError, match="Unrecognized location format"):
            resolve_location("not-a-location")

    def test_resolve_empty_raises(self):
        """Empty string raises ValueError."""
        from src.location.resolver import resolve_location

        with pytest.raises(ValueError, match="empty"):
            resolve_location("")

    def test_is_pin_code(self):
        """Test PIN code detection."""
        from src.location.resolver import _is_pin_code

        assert _is_pin_code("560001") is True
        assert _is_pin_code("110001") is True
        assert _is_pin_code("56000") is False
        assert _is_pin_code("5600012") is False
        assert _is_pin_code("abc123") is False

    def test_is_coordinates(self):
        """Test coordinate format detection."""
        from src.location.resolver import _is_coordinates

        assert _is_coordinates("12.9716,77.5946") is True
        assert _is_coordinates("12.97 , 77.59") is True
        assert _is_coordinates("-33.8,151.2") is True
        assert _is_coordinates("12.97") is False
        assert _is_coordinates("abc,def") is False


# ---------- SoilGrids tests ----------

class TestSoilGrids:
    def test_fetch_soilgrids_success(self):
        """Parse SoilGrids JSON response correctly."""
        from src.location.soilgrids import fetch_soilgrids

        mock_data = {
            "properties": {
                "layers": [
                    {
                        "name": "phh2o",
                        "depths": [{"label": "0-5cm", "values": {"mean": 65}}],
                    },
                    {
                        "name": "clay",
                        "depths": [{"label": "0-5cm", "values": {"mean": 250}}],
                    },
                    {
                        "name": "soc",
                        "depths": [{"label": "0-5cm", "values": {"mean": 120}}],
                    },
                    {
                        "name": "nitrogen",
                        "depths": [{"label": "0-5cm", "values": {"mean": 500}}],
                    },
                    {
                        "name": "cec",
                        "depths": [{"label": "0-5cm", "values": {"mean": 150}}],
                    },
                ],
            },
        }
        with patch("src.location.soilgrids.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_data
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            result = fetch_soilgrids(12.97, 77.59)

        assert result["pH"] == 6.5  # 65 / 10
        assert result["clay_pct"] == 25.0  # 250 / 10
        assert result["soc_g_kg"] == 12.0  # 120 / 10
        assert result["nitrogen_g_kg"] == 5.0  # 500 / 100
        assert result["cec_mmol_kg"] == 150.0

    def test_fetch_soilgrids_api_error(self):
        """API failure returns None values gracefully."""
        from src.location.soilgrids import fetch_soilgrids
        import requests as req_module

        with patch("src.location.soilgrids.requests.get") as mock_get:
            mock_get.side_effect = req_module.exceptions.ConnectionError("down")

            result = fetch_soilgrids(12.97, 77.59)

        assert result["pH"] is None
        assert result["clay_pct"] is None

    def test_pedotransfer_npk(self):
        """Estimate N/P/K from soil properties."""
        from src.location.soilgrids import estimate_npk_from_soil

        soil = {
            "soc_g_kg": 12.0,
            "nitrogen_g_kg": 2.0,
            "cec_mmol_kg": 150.0,
            "clay_pct": 25.0,
        }
        npk = estimate_npk_from_soil(soil)

        assert "N_kg_ha" in npk
        assert "P_kg_ha" in npk
        assert "K_kg_ha" in npk
        assert npk["N_kg_ha"] > 0
        assert npk["P_kg_ha"] > 0
        assert npk["K_kg_ha"] > 0

    def test_pedotransfer_with_none_values(self):
        """Pedotransfer returns defaults when soil data is all None."""
        from src.location.soilgrids import estimate_npk_from_soil

        soil = {k: None for k in ["soc_g_kg", "nitrogen_g_kg", "cec_mmol_kg", "clay_pct"]}
        npk = estimate_npk_from_soil(soil)

        # Should return sensible defaults
        assert npk["N_kg_ha"] == 50.0
        assert npk["P_kg_ha"] == 25.0
        assert npk["K_kg_ha"] == 25.0


# ---------- Weather tests ----------

class TestWeather:
    def test_fetch_weather_success(self):
        """Parse Open-Meteo JSON response correctly."""
        from src.location.weather import fetch_weather

        mock_data = {
            "daily": {
                "temperature_2m_mean": [25.0, 26.0, 27.0],
                "temperature_2m_max": [30.0, 31.0, 32.0],
                "temperature_2m_min": [20.0, 21.0, 22.0],
                "precipitation_sum": [5.0, 0.0, 10.0],
                "relative_humidity_2m_mean": [70.0, 72.0, 68.0],
            },
        }
        with patch("src.location.weather.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_data
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            result = fetch_weather(12.97, 77.59)

        assert result["avg_temp_c"] == 26.0
        assert result["humidity_pct"] == 70.0
        assert result["temp_max_c"] == 31.0
        assert result["data_points"] == 3

    def test_fetch_weather_api_error(self):
        """API failure returns None values gracefully."""
        from src.location.weather import fetch_weather
        import requests as req_module

        with patch("src.location.weather.requests.get") as mock_get:
            mock_get.side_effect = req_module.exceptions.Timeout("timeout")

            result = fetch_weather(12.97, 77.59)

        assert result["avg_temp_c"] is None
        assert result["humidity_pct"] is None
        assert result["data_points"] == 0


# ---------- Market tests ----------

class TestMarket:
    def test_fetch_market_no_api_key(self):
        """Without API key, returns empty dict gracefully."""
        from src.location.market import fetch_market_prices

        with patch.dict("os.environ", {}, clear=True):
            result = fetch_market_prices("Karnataka", ["rice"])

        assert result == {}

    def test_fetch_market_with_key(self):
        """With API key, parses response correctly."""
        from src.location.market import fetch_market_prices

        mock_data = {
            "records": [{
                "Modal_x0020_Price": "2100",
                "Min_x0020_Price": "1900",
                "Max_x0020_Price": "2300",
                "Market": "Bengaluru",
            }],
        }
        with patch.dict("os.environ", {"DATA_GOV_IN_API_KEY": "test_key"}):
            with patch("src.location.market.requests.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.json.return_value = mock_data
                mock_resp.raise_for_status = MagicMock()
                mock_get.return_value = mock_resp

                result = fetch_market_prices("Karnataka", ["rice"])

        assert "rice" in result
        assert result["rice"]["modal_price_per_quintal"] == 2100.0


# ---------- Pipeline orchestrator tests ----------

class TestLocationPipeline:
    def test_tertiary_pipeline_full(self):
        """End-to-end Tertiary pipeline with all mocked HTTP calls."""
        from src.location.pipeline import LocationPipeline, PipelineConfig

        # Mock all the HTTP calls
        with patch("src.location.resolver.requests.get") as mock_resolver, \
             patch("src.location.soilgrids.requests.get") as mock_soil, \
             patch("src.location.weather.requests.get") as mock_weather:

            # Mock reverse geocode
            mock_resolver_resp = MagicMock()
            mock_resolver_resp.json.return_value = {
                "address": {"state": "Karnataka", "state_district": "Bangalore"},
                "display_name": "Bengaluru, Karnataka",
            }
            mock_resolver_resp.raise_for_status = MagicMock()
            mock_resolver.return_value = mock_resolver_resp

            # Mock SoilGrids
            mock_soil_resp = MagicMock()
            mock_soil_resp.json.return_value = {
                "properties": {
                    "layers": [
                        {"name": "phh2o", "depths": [{"label": "0-5cm", "values": {"mean": 65}}]},
                        {"name": "clay", "depths": [{"label": "0-5cm", "values": {"mean": 300}}]},
                        {"name": "soc", "depths": [{"label": "0-5cm", "values": {"mean": 100}}]},
                        {"name": "nitrogen", "depths": [{"label": "0-5cm", "values": {"mean": 400}}]},
                        {"name": "cec", "depths": [{"label": "0-5cm", "values": {"mean": 120}}]},
                    ],
                },
            }
            mock_soil_resp.raise_for_status = MagicMock()
            mock_soil.return_value = mock_soil_resp

            # Mock weather
            mock_weather_resp = MagicMock()
            mock_weather_resp.json.return_value = {
                "daily": {
                    "temperature_2m_mean": [28.0] * 365,
                    "temperature_2m_max": [33.0] * 365,
                    "temperature_2m_min": [23.0] * 365,
                    "precipitation_sum": [3.0] * 365,
                    "relative_humidity_2m_mean": [72.0] * 365,
                },
            }
            mock_weather_resp.raise_for_status = MagicMock()
            mock_weather.return_value = mock_weather_resp

            config = PipelineConfig(
                location="12.9716,77.5946",
                pipeline="Tertiary",
            )
            pipeline = LocationPipeline()
            result = pipeline.run(config)

        # Verify structure
        assert result.pipeline_used == "Tertiary"
        assert result.location_info["latitude"] == 12.9716
        assert result.soil_data["pH"] == 6.5
        assert result.weather_data["avg_temp_c"] == 28.0
        assert "N" in result.model_input
        assert "temperature" in result.model_input
        assert "rainfall" in result.model_input
        assert len(result.data_sources) >= 2
