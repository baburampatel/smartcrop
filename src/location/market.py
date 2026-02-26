"""
Agmarknet market price client: fetches commodity prices from data.gov.in.
Requires a free API key (set DATA_GOV_IN_API_KEY env var).
Optional module -- degrades gracefully if API key is not set.

API docs: https://data.gov.in/ogpl_apis
License: GODL (Government Open Data License - India)
"""

import os
import logging
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# Resource ID for daily commodity price data on data.gov.in
AGMARKNET_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
DATA_GOV_API_BASE = "https://api.data.gov.in/resource"

# Map our canonical crop names to Agmarknet commodity names
CROP_TO_COMMODITY = {
    "rice": "Rice",
    "maize": "Maize",
    "wheat": "Wheat",
    "soybean": "Soyabean",
    "groundnut": "Groundnut",
    "sugarcane": "Sugarcane",
    "chickpea": "Bengal Gram(Gram)(Whole)",
    "lentil": "Masur Dal",
    "cotton": "Cotton",
    "jute": "Jute",
    "coffee": "Coffee",
    "banana": "Banana",
    "mango": "Mango",
    "coconut": "Coconut",
    "orange": "Orange",
    "papaya": "Papaya",
    "grapes": "Grapes",
}


def _get_api_key() -> Optional[str]:
    """Get data.gov.in API key from environment."""
    return os.environ.get("DATA_GOV_IN_API_KEY")


def fetch_market_prices(
    state: str,
    crops: Optional[List[str]] = None,
    limit: int = 50,
) -> Dict[str, Dict]:
    """
    Fetch recent market (mandi) prices for crops in a given state.

    Args:
        state: Indian state name (e.g., 'Karnataka')
        crops: List of canonical crop names to query (default: all known)
        limit: Max records per API call

    Returns:
        Dict mapping crop name -> {
            'modal_price_per_quintal': float,
            'min_price_per_quintal': float,
            'max_price_per_quintal': float,
            'market': str,
            'commodity': str,
        }
        Empty dict if API key is not set or API fails.
    """
    api_key = _get_api_key()
    if not api_key:
        logger.info(
            "DATA_GOV_IN_API_KEY not set; skipping market price fetch. "
            "Register for a free key at https://data.gov.in/user/register"
        )
        return {}

    if crops is None:
        crops = list(CROP_TO_COMMODITY.keys())

    result = {}
    for crop in crops:
        commodity = CROP_TO_COMMODITY.get(crop.lower())
        if not commodity:
            continue

        url = f"{DATA_GOV_API_BASE}/{AGMARKNET_RESOURCE_ID}"
        params = {
            "api-key": api_key,
            "format": "json",
            "limit": limit,
            "filters[State.keyword]": state,
            "filters[Commodity]": commodity,
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            logger.warning("Agmarknet API failed for %s: %s", crop, e)
            continue

        records = data.get("records", [])
        if not records:
            continue

        # Take the most recent record with valid prices
        for record in records:
            try:
                modal = float(record.get("Modal_x0020_Price", 0))
                min_p = float(record.get("Min_x0020_Price", 0))
                max_p = float(record.get("Max_x0020_Price", 0))
                if modal > 0:
                    result[crop] = {
                        "modal_price_per_quintal": modal,
                        "min_price_per_quintal": min_p,
                        "max_price_per_quintal": max_p,
                        "market": record.get("Market", ""),
                        "commodity": commodity,
                    }
                    break
            except (ValueError, TypeError):
                continue

    return result
