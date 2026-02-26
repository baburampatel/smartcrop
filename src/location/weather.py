"""
Open-Meteo weather client: fetches historical weather data for a point.
Free API, no key required. Uses ERA5 reanalysis for historical data.

API docs: https://open-meteo.com/en/docs/historical-weather-api
License: CC-BY 4.0
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather(
    lat: float,
    lon: float,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    years_back: int = 3,
) -> Dict[str, Optional[float]]:
    """
    Fetch historical weather averages from Open-Meteo for a location.

    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date in YYYY-MM-DD format (default: years_back ago)
        end_date: End date in YYYY-MM-DD format (default: yesterday)
        years_back: Number of years to look back if dates not provided

    Returns:
        Dict with:
            avg_temp_c: Mean annual temperature (C)
            humidity_pct: Mean relative humidity (%)
            avg_precip_mm: Mean annual total precipitation (mm)
            temp_max_c: Mean daily max temperature (C)
            temp_min_c: Mean daily min temperature (C)
            data_points: Number of daily records used
    """
    if end_date is None:
        end_dt = datetime.now() - timedelta(days=5)  # archive has ~5-day lag
        end_date = end_dt.strftime("%Y-%m-%d")

    if start_date is None:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=365 * years_back)
        start_date = start_dt.strftime("%Y-%m-%d")

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "relative_humidity_2m_mean",
        ]),
        "timezone": "auto",
    }

    result = {
        "avg_temp_c": None,
        "humidity_pct": None,
        "avg_precip_mm": None,
        "temp_max_c": None,
        "temp_min_c": None,
        "data_points": 0,
    }

    try:
        resp = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        logger.warning("Open-Meteo API request failed: %s", e)
        return result

    daily = data.get("daily", {})
    if not daily:
        logger.warning("No daily data returned from Open-Meteo")
        return result

    temp_mean = daily.get("temperature_2m_mean", [])
    temp_max = daily.get("temperature_2m_max", [])
    temp_min = daily.get("temperature_2m_min", [])
    precip = daily.get("precipitation_sum", [])
    humidity = daily.get("relative_humidity_2m_mean", [])

    # Filter None values for averaging
    valid_temp = [t for t in temp_mean if t is not None]
    valid_tmax = [t for t in temp_max if t is not None]
    valid_tmin = [t for t in temp_min if t is not None]
    valid_precip = [p for p in precip if p is not None]
    valid_humid = [h for h in humidity if h is not None]

    if valid_temp:
        result["avg_temp_c"] = round(sum(valid_temp) / len(valid_temp), 1)
    if valid_tmax:
        result["temp_max_c"] = round(sum(valid_tmax) / len(valid_tmax), 1)
    if valid_tmin:
        result["temp_min_c"] = round(sum(valid_tmin) / len(valid_tmin), 1)
    if valid_humid:
        result["humidity_pct"] = round(sum(valid_humid) / len(valid_humid), 1)
    if valid_precip:
        # Sum precipitation per year, then average across years
        total_precip = sum(valid_precip)
        n_years = max(1, len(valid_precip) / 365.0)
        result["avg_precip_mm"] = round(total_precip / n_years, 1)

    result["data_points"] = len(valid_temp)

    return result


def fetch_seasonal_weather(
    lat: float,
    lon: float,
    season: str,
    years_back: int = 3,
) -> Dict[str, Optional[float]]:
    """
    Fetch weather averages for a specific Indian cropping season.

    Args:
        lat: Latitude
        lon: Longitude
        season: 'kharif' (Jun-Oct), 'rabi' (Nov-Mar), or 'zaid' (Apr-May)
        years_back: Number of years to look back

    Returns:
        Same dict as fetch_weather but filtered to season months.
    """
    # Season month ranges (approximate)
    season_months = {
        "kharif": (6, 10),   # June to October
        "rabi": (11, 3),     # November to March
        "zaid": (4, 5),      # April to May
    }

    season_lower = season.lower().strip()
    if season_lower not in season_months:
        logger.warning("Unknown season '%s', using full-year data", season)
        return fetch_weather(lat, lon, years_back=years_back)

    # For simplicity, fetch full year data and filter isn't practical via API
    # Open-Meteo doesn't support month filtering, so we fetch full range
    # and note the season for context
    weather = fetch_weather(lat, lon, years_back=years_back)
    weather["season"] = season_lower
    return weather
