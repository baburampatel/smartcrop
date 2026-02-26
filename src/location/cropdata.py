"""
Regional crop data: what crops are currently grown at a given location.

Uses two approaches:
1. India Crop Statistics — curated state/season crop data from ICAR/MoA
2. NDVI vegetation index — fetched from NASA POWER API (free, no key needed)
   to indicate current growing activity.

Future: integrate Sentinel-2 or ISRO Bhuvan crop-type classification maps.
"""

import logging
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# NASA POWER API for agroclimatology (free, no key)
NASA_POWER_API = "https://power.larc.nasa.gov/api/temporal/monthly"

# ---- Curated Indian crop statistics by state ----
# Source: Directorate of Economics & Statistics, Ministry of Agriculture, India
# Each state maps to a list of (crop, season, relative_area_rank)
STATE_CROPS: Dict[str, List[Dict]] = {
    "Andhra Pradesh": [
        {"crop": "rice", "season": "kharif", "rank": 1},
        {"crop": "groundnut", "season": "kharif", "rank": 2},
        {"crop": "sugarcane", "season": "annual", "rank": 3},
        {"crop": "maize", "season": "kharif", "rank": 4},
        {"crop": "cotton", "season": "kharif", "rank": 5},
    ],
    "Assam": [
        {"crop": "rice", "season": "kharif", "rank": 1},
        {"crop": "jute", "season": "kharif", "rank": 2},
        {"crop": "tea", "season": "annual", "rank": 3},
        {"crop": "sugarcane", "season": "annual", "rank": 4},
    ],
    "Bihar": [
        {"crop": "rice", "season": "kharif", "rank": 1},
        {"crop": "wheat", "season": "rabi", "rank": 2},
        {"crop": "maize", "season": "kharif", "rank": 3},
        {"crop": "sugarcane", "season": "annual", "rank": 4},
        {"crop": "lentil", "season": "rabi", "rank": 5},
    ],
    "Chhattisgarh": [
        {"crop": "rice", "season": "kharif", "rank": 1},
        {"crop": "soybean", "season": "kharif", "rank": 2},
        {"crop": "wheat", "season": "rabi", "rank": 3},
        {"crop": "maize", "season": "kharif", "rank": 4},
    ],
    "Goa": [
        {"crop": "rice", "season": "kharif", "rank": 1},
        {"crop": "coconut", "season": "annual", "rank": 2},
        {"crop": "cashew", "season": "annual", "rank": 3},
    ],
    "Gujarat": [
        {"crop": "groundnut", "season": "kharif", "rank": 1},
        {"crop": "cotton", "season": "kharif", "rank": 2},
        {"crop": "wheat", "season": "rabi", "rank": 3},
        {"crop": "rice", "season": "kharif", "rank": 4},
        {"crop": "sugarcane", "season": "annual", "rank": 5},
    ],
    "Haryana": [
        {"crop": "wheat", "season": "rabi", "rank": 1},
        {"crop": "rice", "season": "kharif", "rank": 2},
        {"crop": "sugarcane", "season": "annual", "rank": 3},
        {"crop": "cotton", "season": "kharif", "rank": 4},
        {"crop": "mustard", "season": "rabi", "rank": 5},
    ],
    "Himachal Pradesh": [
        {"crop": "wheat", "season": "rabi", "rank": 1},
        {"crop": "maize", "season": "kharif", "rank": 2},
        {"crop": "rice", "season": "kharif", "rank": 3},
        {"crop": "apple", "season": "annual", "rank": 4},
    ],
    "Jharkhand": [
        {"crop": "rice", "season": "kharif", "rank": 1},
        {"crop": "wheat", "season": "rabi", "rank": 2},
        {"crop": "maize", "season": "kharif", "rank": 3},
        {"crop": "lentil", "season": "rabi", "rank": 4},
    ],
    "Karnataka": [
        {"crop": "rice", "season": "kharif", "rank": 1},
        {"crop": "maize", "season": "kharif", "rank": 2},
        {"crop": "coffee", "season": "annual", "rank": 3},
        {"crop": "sugarcane", "season": "annual", "rank": 4},
        {"crop": "groundnut", "season": "kharif", "rank": 5},
        {"crop": "cotton", "season": "kharif", "rank": 6},
    ],
    "Kerala": [
        {"crop": "rice", "season": "kharif", "rank": 1},
        {"crop": "coconut", "season": "annual", "rank": 2},
        {"crop": "coffee", "season": "annual", "rank": 3},
        {"crop": "banana", "season": "annual", "rank": 4},
        {"crop": "rubber", "season": "annual", "rank": 5},
    ],
    "Madhya Pradesh": [
        {"crop": "soybean", "season": "kharif", "rank": 1},
        {"crop": "wheat", "season": "rabi", "rank": 2},
        {"crop": "chickpea", "season": "rabi", "rank": 3},
        {"crop": "rice", "season": "kharif", "rank": 4},
        {"crop": "maize", "season": "kharif", "rank": 5},
    ],
    "Maharashtra": [
        {"crop": "sugarcane", "season": "annual", "rank": 1},
        {"crop": "soybean", "season": "kharif", "rank": 2},
        {"crop": "cotton", "season": "kharif", "rank": 3},
        {"crop": "rice", "season": "kharif", "rank": 4},
        {"crop": "groundnut", "season": "kharif", "rank": 5},
        {"crop": "wheat", "season": "rabi", "rank": 6},
    ],
    "Manipur": [
        {"crop": "rice", "season": "kharif", "rank": 1},
        {"crop": "maize", "season": "kharif", "rank": 2},
    ],
    "Meghalaya": [
        {"crop": "rice", "season": "kharif", "rank": 1},
        {"crop": "maize", "season": "kharif", "rank": 2},
        {"crop": "potato", "season": "rabi", "rank": 3},
    ],
    "Mizoram": [
        {"crop": "rice", "season": "kharif", "rank": 1},
        {"crop": "maize", "season": "kharif", "rank": 2},
    ],
    "Nagaland": [
        {"crop": "rice", "season": "kharif", "rank": 1},
        {"crop": "maize", "season": "kharif", "rank": 2},
    ],
    "Odisha": [
        {"crop": "rice", "season": "kharif", "rank": 1},
        {"crop": "groundnut", "season": "kharif", "rank": 2},
        {"crop": "sugarcane", "season": "annual", "rank": 3},
        {"crop": "maize", "season": "kharif", "rank": 4},
    ],
    "Punjab": [
        {"crop": "wheat", "season": "rabi", "rank": 1},
        {"crop": "rice", "season": "kharif", "rank": 2},
        {"crop": "cotton", "season": "kharif", "rank": 3},
        {"crop": "maize", "season": "kharif", "rank": 4},
        {"crop": "sugarcane", "season": "annual", "rank": 5},
    ],
    "Rajasthan": [
        {"crop": "wheat", "season": "rabi", "rank": 1},
        {"crop": "mustard", "season": "rabi", "rank": 2},
        {"crop": "groundnut", "season": "kharif", "rank": 3},
        {"crop": "maize", "season": "kharif", "rank": 4},
        {"crop": "chickpea", "season": "rabi", "rank": 5},
    ],
    "Sikkim": [
        {"crop": "rice", "season": "kharif", "rank": 1},
        {"crop": "maize", "season": "kharif", "rank": 2},
        {"crop": "cardamom", "season": "annual", "rank": 3},
    ],
    "Tamil Nadu": [
        {"crop": "rice", "season": "kharif", "rank": 1},
        {"crop": "sugarcane", "season": "annual", "rank": 2},
        {"crop": "groundnut", "season": "kharif", "rank": 3},
        {"crop": "cotton", "season": "kharif", "rank": 4},
        {"crop": "banana", "season": "annual", "rank": 5},
        {"crop": "coconut", "season": "annual", "rank": 6},
    ],
    "Telangana": [
        {"crop": "rice", "season": "kharif", "rank": 1},
        {"crop": "cotton", "season": "kharif", "rank": 2},
        {"crop": "maize", "season": "kharif", "rank": 3},
        {"crop": "soybean", "season": "kharif", "rank": 4},
        {"crop": "sugarcane", "season": "annual", "rank": 5},
    ],
    "Tripura": [
        {"crop": "rice", "season": "kharif", "rank": 1},
        {"crop": "jute", "season": "kharif", "rank": 2},
    ],
    "Uttar Pradesh": [
        {"crop": "wheat", "season": "rabi", "rank": 1},
        {"crop": "rice", "season": "kharif", "rank": 2},
        {"crop": "sugarcane", "season": "annual", "rank": 3},
        {"crop": "potato", "season": "rabi", "rank": 4},
        {"crop": "mustard", "season": "rabi", "rank": 5},
        {"crop": "maize", "season": "kharif", "rank": 6},
    ],
    "Uttarakhand": [
        {"crop": "rice", "season": "kharif", "rank": 1},
        {"crop": "wheat", "season": "rabi", "rank": 2},
        {"crop": "sugarcane", "season": "annual", "rank": 3},
        {"crop": "soybean", "season": "kharif", "rank": 4},
    ],
    "West Bengal": [
        {"crop": "rice", "season": "kharif", "rank": 1},
        {"crop": "jute", "season": "kharif", "rank": 2},
        {"crop": "wheat", "season": "rabi", "rank": 3},
        {"crop": "potato", "season": "rabi", "rank": 4},
        {"crop": "sugarcane", "season": "annual", "rank": 5},
    ],
    # Union territories
    "Delhi": [
        {"crop": "wheat", "season": "rabi", "rank": 1},
        {"crop": "rice", "season": "kharif", "rank": 2},
    ],
    "National Capital Territory of Delhi": [
        {"crop": "wheat", "season": "rabi", "rank": 1},
        {"crop": "rice", "season": "kharif", "rank": 2},
    ],
    "Jammu and Kashmir": [
        {"crop": "rice", "season": "kharif", "rank": 1},
        {"crop": "wheat", "season": "rabi", "rank": 2},
        {"crop": "maize", "season": "kharif", "rank": 3},
        {"crop": "apple", "season": "annual", "rank": 4},
    ],
}

# Season emoji/label for display
SEASON_LABELS = {
    "kharif": "Kharif (Jun-Oct)",
    "rabi": "Rabi (Nov-Mar)",
    "zaid": "Zaid (Apr-May)",
    "annual": "Year-round",
}


def get_regional_crops(state: Optional[str]) -> List[Dict]:
    """
    Get crops commonly grown in a state from curated agriculture statistics.

    Args:
        state: Indian state name (e.g., 'Karnataka')

    Returns:
        List of dicts with crop, season, rank, season_label.
        Empty list if state not found.
    """
    if not state:
        return []

    # Try exact match first, then case-insensitive
    crops = STATE_CROPS.get(state)
    if crops is None:
        for key, val in STATE_CROPS.items():
            if key.lower() == state.lower():
                crops = val
                break

    if crops is None:
        logger.info("No crop data for state: %s", state)
        return []

    # Add readable season labels
    result = []
    for c in crops:
        entry = c.copy()
        entry["season_label"] = SEASON_LABELS.get(c["season"], c["season"])
        result.append(entry)

    return result


def fetch_ndvi_indicator(lat: float, lon: float) -> Optional[Dict]:
    """
    Fetch vegetation index (NDVI proxy) from NASA POWER API.
    Uses T2M_RANGE and ALLSKY_SFC_SW_DWN as vegetation activity indicators.
    Free API, no key needed.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        Dict with vegetation_activity estimate, or None on failure.
    """
    try:
        # Fetch recent months of vegetation-related parameters
        params = {
            "parameters": "T2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN",
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": "2024",
            "end": "2024",
            "format": "JSON",
        }
        resp = requests.get(
            NASA_POWER_API, params=params, timeout=15
        )
        resp.raise_for_status()
        data = resp.json()

        properties = data.get("properties", {}).get("parameter", {})
        precip = properties.get("PRECTOTCORR", {})
        solar = properties.get("ALLSKY_SFC_SW_DWN", {})

        # Calculate average monthly precipitation and solar radiation
        precip_vals = [v for v in precip.values() if v > 0]
        solar_vals = [v for v in solar.values() if v > 0]

        avg_precip = sum(precip_vals) / len(precip_vals) if precip_vals else 0
        avg_solar = sum(solar_vals) / len(solar_vals) if solar_vals else 0

        # Simple vegetation activity score (0-100)
        # Higher precip + higher solar = more vegetation activity
        veg_score = min(100, (avg_precip / 5.0) * 40 + (avg_solar / 6.0) * 60)

        if veg_score > 70:
            activity = "High"
        elif veg_score > 40:
            activity = "Moderate"
        else:
            activity = "Low"

        return {
            "vegetation_activity": activity,
            "vegetation_score": round(veg_score, 1),
            "avg_monthly_precip_mm_day": round(avg_precip, 2),
            "avg_solar_radiation_kwh_m2_day": round(avg_solar, 2),
            "source": "NASA POWER (satellite-derived)",
        }

    except Exception as e:
        logger.warning("NASA POWER API failed: %s", e)
        return None
