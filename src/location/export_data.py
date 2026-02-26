"""
India agricultural export statistics — curated from APEDA ExportStat,
Ministry of Commerce, and DGCI&S public reports (FY 2024-25).

Provides export volume, value, growth rates, destinations, and seasonal
windows for major Indian agricultural commodities.
"""

from typing import Dict, List, Optional

# ---- Curated India Agri-Export Data (FY 2024-25) ----
# Source: APEDA ExportStat, Ministry of Commerce annual trade data
# All values are approximate annualized figures

EXPORT_STATS: Dict[str, Dict] = {
    "rice": {
        "export_volume_mt": 17_800_000,      # metric tons
        "export_value_usd_mn": 10_200,       # million USD
        "quarterly_growth_pct": 5.2,         # Q3 vs Q2 growth
        "avg_price_usd_ton": 573,
        "price_trend": "stable",             # rising | stable | falling
        "top_destinations": ["Bangladesh", "Saudi Arabia", "Iran", "Iraq", "Nepal"],
        "peak_export_months": [10, 11, 12, 1, 2],  # Oct-Feb (post-kharif harvest)
        "harvest_to_export_weeks": 4,
        "cold_chain_required": False,
        "processing_required": "milling",
        "apeda_registered": True,
        "phytosanitary_complexity": "low",    # low | medium | high
        "certification_days": 7,
    },
    "wheat": {
        "export_volume_mt": 2_100_000,
        "export_value_usd_mn": 780,
        "quarterly_growth_pct": -12.5,
        "avg_price_usd_ton": 371,
        "price_trend": "falling",
        "top_destinations": ["Bangladesh", "Sri Lanka", "UAE", "Indonesia"],
        "peak_export_months": [4, 5, 6, 7],
        "harvest_to_export_weeks": 3,
        "cold_chain_required": False,
        "processing_required": "cleaning",
        "apeda_registered": True,
        "phytosanitary_complexity": "low",
        "certification_days": 7,
    },
    "maize": {
        "export_volume_mt": 3_500_000,
        "export_value_usd_mn": 1_050,
        "quarterly_growth_pct": 8.7,
        "avg_price_usd_ton": 300,
        "price_trend": "rising",
        "top_destinations": ["Vietnam", "Bangladesh", "Nepal", "Malaysia", "Indonesia"],
        "peak_export_months": [11, 12, 1, 2, 3],
        "harvest_to_export_weeks": 3,
        "cold_chain_required": False,
        "processing_required": "drying",
        "apeda_registered": True,
        "phytosanitary_complexity": "low",
        "certification_days": 5,
    },
    "soybean": {
        "export_volume_mt": 450_000,
        "export_value_usd_mn": 280,
        "quarterly_growth_pct": 3.1,
        "avg_price_usd_ton": 622,
        "price_trend": "stable",
        "top_destinations": ["Japan", "South Korea", "Thailand", "Vietnam"],
        "peak_export_months": [11, 12, 1, 2],
        "harvest_to_export_weeks": 4,
        "cold_chain_required": False,
        "processing_required": "cleaning",
        "apeda_registered": True,
        "phytosanitary_complexity": "medium",
        "certification_days": 10,
    },
    "groundnut": {
        "export_volume_mt": 680_000,
        "export_value_usd_mn": 920,
        "quarterly_growth_pct": 14.3,
        "avg_price_usd_ton": 1_353,
        "price_trend": "rising",
        "top_destinations": ["Vietnam", "Indonesia", "Philippines", "Malaysia", "China"],
        "peak_export_months": [11, 12, 1, 2, 3],
        "harvest_to_export_weeks": 5,
        "cold_chain_required": False,
        "processing_required": "shelling, sorting",
        "apeda_registered": True,
        "phytosanitary_complexity": "medium",
        "certification_days": 10,
    },
    "sugarcane": {
        "export_volume_mt": 5_200_000,   # as sugar
        "export_value_usd_mn": 2_800,
        "quarterly_growth_pct": -3.4,
        "avg_price_usd_ton": 538,
        "price_trend": "stable",
        "top_destinations": ["UAE", "Bangladesh", "Sri Lanka", "Indonesia", "Somalia"],
        "peak_export_months": [3, 4, 5, 6, 7],
        "harvest_to_export_weeks": 8,    # needs processing into sugar
        "cold_chain_required": False,
        "processing_required": "sugar mill processing",
        "apeda_registered": True,
        "phytosanitary_complexity": "low",
        "certification_days": 7,
    },
    "cotton": {
        "export_volume_mt": 1_200_000,
        "export_value_usd_mn": 2_100,
        "quarterly_growth_pct": 6.8,
        "avg_price_usd_ton": 1_750,
        "price_trend": "rising",
        "top_destinations": ["Bangladesh", "China", "Vietnam", "Indonesia"],
        "peak_export_months": [11, 12, 1, 2, 3],
        "harvest_to_export_weeks": 6,
        "cold_chain_required": False,
        "processing_required": "ginning",
        "apeda_registered": False,
        "phytosanitary_complexity": "low",
        "certification_days": 5,
    },
    "coffee": {
        "export_volume_mt": 400_000,
        "export_value_usd_mn": 1_250,
        "quarterly_growth_pct": 11.2,
        "avg_price_usd_ton": 3_125,
        "price_trend": "rising",
        "top_destinations": ["Italy", "Germany", "Belgium", "Russia", "Turkey"],
        "peak_export_months": [1, 2, 3, 4, 5],
        "harvest_to_export_weeks": 8,
        "cold_chain_required": False,
        "processing_required": "pulping, drying, grading",
        "apeda_registered": True,
        "phytosanitary_complexity": "medium",
        "certification_days": 14,
    },
    "banana": {
        "export_volume_mt": 220_000,
        "export_value_usd_mn": 180,
        "quarterly_growth_pct": 18.5,
        "avg_price_usd_ton": 818,
        "price_trend": "rising",
        "top_destinations": ["UAE", "Saudi Arabia", "Bahrain", "Iran", "Iraq"],
        "peak_export_months": [1, 2, 3, 4, 10, 11, 12],
        "harvest_to_export_weeks": 2,
        "cold_chain_required": True,
        "processing_required": "ripening rooms, packing",
        "apeda_registered": True,
        "phytosanitary_complexity": "high",
        "certification_days": 14,
    },
    "coconut": {
        "export_volume_mt": 150_000,
        "export_value_usd_mn": 350,
        "quarterly_growth_pct": 7.2,
        "avg_price_usd_ton": 2_333,
        "price_trend": "rising",
        "top_destinations": ["UAE", "USA", "UK", "Netherlands"],
        "peak_export_months": list(range(1, 13)),  # year-round
        "harvest_to_export_weeks": 3,
        "cold_chain_required": False,
        "processing_required": "drying, copra processing",
        "apeda_registered": True,
        "phytosanitary_complexity": "low",
        "certification_days": 7,
    },
    "chickpea": {
        "export_volume_mt": 180_000,
        "export_value_usd_mn": 140,
        "quarterly_growth_pct": 4.5,
        "avg_price_usd_ton": 778,
        "price_trend": "stable",
        "top_destinations": ["UAE", "Algeria", "Turkey", "Pakistan"],
        "peak_export_months": [4, 5, 6, 7],
        "harvest_to_export_weeks": 3,
        "cold_chain_required": False,
        "processing_required": "cleaning, grading",
        "apeda_registered": True,
        "phytosanitary_complexity": "low",
        "certification_days": 7,
    },
    "mustard": {
        "export_volume_mt": 95_000,
        "export_value_usd_mn": 85,
        "quarterly_growth_pct": 2.1,
        "avg_price_usd_ton": 895,
        "price_trend": "stable",
        "top_destinations": ["Nepal", "Bangladesh", "UAE"],
        "peak_export_months": [4, 5, 6],
        "harvest_to_export_weeks": 3,
        "cold_chain_required": False,
        "processing_required": "oil extraction or seed cleaning",
        "apeda_registered": True,
        "phytosanitary_complexity": "low",
        "certification_days": 7,
    },
    "potato": {
        "export_volume_mt": 350_000,
        "export_value_usd_mn": 120,
        "quarterly_growth_pct": 9.3,
        "avg_price_usd_ton": 343,
        "price_trend": "stable",
        "top_destinations": ["Nepal", "Sri Lanka", "Malaysia", "UAE"],
        "peak_export_months": [2, 3, 4, 5],
        "harvest_to_export_weeks": 2,
        "cold_chain_required": True,
        "processing_required": "grading, cold storage",
        "apeda_registered": True,
        "phytosanitary_complexity": "medium",
        "certification_days": 10,
    },
    "jute": {
        "export_volume_mt": 200_000,
        "export_value_usd_mn": 160,
        "quarterly_growth_pct": -1.5,
        "avg_price_usd_ton": 800,
        "price_trend": "falling",
        "top_destinations": ["Bangladesh", "China", "Thailand"],
        "peak_export_months": [9, 10, 11, 12],
        "harvest_to_export_weeks": 6,
        "cold_chain_required": False,
        "processing_required": "retting, baling",
        "apeda_registered": False,
        "phytosanitary_complexity": "low",
        "certification_days": 5,
    },
    "lentil": {
        "export_volume_mt": 120_000,
        "export_value_usd_mn": 110,
        "quarterly_growth_pct": 5.8,
        "avg_price_usd_ton": 917,
        "price_trend": "stable",
        "top_destinations": ["Sri Lanka", "UAE", "Bangladesh"],
        "peak_export_months": [4, 5, 6],
        "harvest_to_export_weeks": 3,
        "cold_chain_required": False,
        "processing_required": "cleaning, grading",
        "apeda_registered": True,
        "phytosanitary_complexity": "low",
        "certification_days": 7,
    },
}


def get_export_stats(crop: str) -> Optional[Dict]:
    """
    Get export statistics for a crop.

    Args:
        crop: Crop name (case-insensitive).

    Returns:
        Dict with export stats, or None if crop not found.
    """
    return EXPORT_STATS.get(crop.lower())


def get_all_export_crops() -> List[str]:
    """Return all crops with export data available."""
    return list(EXPORT_STATS.keys())


def get_top_export_crops(n: int = 10, sort_by: str = "value") -> List[Dict]:
    """
    Get top N exported crops from India.

    Args:
        n: Number of crops to return.
        sort_by: 'value' (USD), 'volume' (MT), or 'growth' (quarterly %).

    Returns:
        List of dicts with crop name and stats, sorted descending.
    """
    key_map = {
        "value": "export_value_usd_mn",
        "volume": "export_volume_mt",
        "growth": "quarterly_growth_pct",
    }
    sort_key = key_map.get(sort_by, "export_value_usd_mn")

    ranked = []
    for crop, stats in EXPORT_STATS.items():
        ranked.append({"crop": crop, **stats})

    ranked.sort(key=lambda x: x.get(sort_key, 0), reverse=True)
    return ranked[:n]
