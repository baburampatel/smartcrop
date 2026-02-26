"""
Export scorer: computes an export_score (0–100) for each candidate crop
and selects the best crop to export from a given location.

Score components:
  - Demand momentum (30%): quarterly growth in export volume
  - Price trend (20%): recent price direction
  - Logistics (20%): distance to nearest port, cold chain
  - Timing alignment (20%): harvest-to-export fits EXPORT_WINDOW
  - Regulatory ease (10%): certification complexity

Produces a single `export_suggestion` with human-readable reason.
"""

import datetime
import logging
from typing import Dict, List, Optional

from src.location.export_data import get_export_stats, EXPORT_STATS
from src.location.trade_routes import (
    find_nearest_ports,
    get_logistics_score,
    get_certification_info,
)

logger = logging.getLogger(__name__)

# Score weights
W_DEMAND = 0.30
W_PRICE = 0.20
W_LOGISTICS = 0.20
W_TIMING = 0.20
W_REGULATORY = 0.10


def _demand_score(stats: Dict) -> float:
    """Score 0–100 based on quarterly growth percentage."""
    growth = stats.get("quarterly_growth_pct", 0)
    # Map: -20% → 0, 0% → 40, +20% → 100
    score = 40 + growth * 3
    return max(0, min(100, score))


def _price_score(stats: Dict) -> float:
    """Score 0–100 based on price trend."""
    trend = stats.get("price_trend", "stable")
    trend_scores = {"rising": 85, "stable": 50, "falling": 15}
    return trend_scores.get(trend, 50)


def _timing_score(stats: Dict, target_month: int) -> float:
    """
    Score 0–100 based on whether harvest + processing aligns with export window.
    Higher if target_month falls within peak export months.
    """
    peak = stats.get("peak_export_months", [])
    if not peak:
        return 50  # neutral

    # Check if target month (or +/- 1 month) is in peak
    if target_month in peak:
        return 90
    elif (target_month % 12 + 1) in peak or ((target_month - 2) % 12 + 1) in peak:
        return 60
    else:
        return 20


def _regulatory_score(stats: Dict) -> float:
    """Score 0–100 based on certification complexity."""
    complexity = stats.get("phytosanitary_complexity", "low")
    scores = {"low": 90, "medium": 60, "high": 30}
    return scores.get(complexity, 60)


def compute_export_scores(
    crops: List[str],
    lat: float,
    lon: float,
    export_window: str = "next_quarter",
) -> List[Dict]:
    """
    Compute export scores for candidate crops at a location.

    Args:
        crops: List of crop names to evaluate.
        lat: Latitude of the location.
        lon: Longitude of the location.
        export_window: 'next_quarter' (default) — target export timing.

    Returns:
        List of dicts sorted by export_score (descending), each containing:
        crop, export_score, demand_score, price_score, logistics_score,
        timing_score, regulatory_score, reason, nearest_port, confidence.
    """
    # Determine target month based on export window
    now = datetime.datetime.now()
    if export_window == "next_quarter":
        # Next quarter start month
        current_quarter = (now.month - 1) // 3 + 1
        next_quarter_start = (current_quarter * 3) % 12 + 1
        target_month = next_quarter_start
    else:
        target_month = now.month

    # Get logistics for location
    nearest_ports = find_nearest_ports(lat, lon, n=1)
    nearest_port = nearest_ports[0] if nearest_ports else None

    results = []
    for crop in crops:
        crop_lower = crop.lower().strip()
        stats = get_export_stats(crop_lower)

        if stats is None:
            # No export data for this crop
            results.append({
                "crop": crop_lower,
                "export_score": 0,
                "reason": f"No export data available for {crop_lower}",
                "nearest_port": nearest_port["name"] if nearest_port else "Unknown",
                "confidence": 0,
            })
            continue

        # Compute component scores
        demand = _demand_score(stats)
        price = _price_score(stats)
        logistics_info = get_logistics_score(
            lat, lon, stats.get("cold_chain_required", False)
        )
        logistics = logistics_info["score"]
        timing = _timing_score(stats, target_month)
        regulatory = _regulatory_score(stats)

        # Weighted total
        total = (
            W_DEMAND * demand +
            W_PRICE * price +
            W_LOGISTICS * logistics +
            W_TIMING * timing +
            W_REGULATORY * regulatory
        )

        # Build human-readable reason
        reason_parts = []

        growth = stats.get("quarterly_growth_pct", 0)
        if growth > 5:
            reason_parts.append(
                f"Strong export demand (+{growth}% q/q)"
            )
        elif growth > 0:
            reason_parts.append(
                f"Positive export demand (+{growth}% q/q)"
            )
        else:
            reason_parts.append(
                f"Declining export demand ({growth}% q/q)"
            )

        dests = stats.get("top_destinations", [])[:3]
        if dests:
            reason_parts.append(f"Key markets: {', '.join(dests)}")

        if nearest_port:
            reason_parts.append(
                f"Port: {nearest_port['name']} ({nearest_port['distance_km']:.0f} km, "
                f"~{nearest_port['est_transit_hours']:.0f} hrs)"
            )

        price_usd = stats.get("avg_price_usd_ton", 0)
        if price_usd:
            trend = stats.get("price_trend", "stable")
            reason_parts.append(f"Export price: ${price_usd}/ton ({trend})")

        weeks = stats.get("harvest_to_export_weeks", 0)
        if weeks:
            reason_parts.append(f"Harvest to export: ~{weeks} weeks")

        # Confidence based on data quality
        confidence = min(95, total * 0.95)  # Scale to max 95%

        results.append({
            "crop": crop_lower,
            "export_score": round(total, 1),
            "demand_score": round(demand, 1),
            "price_score": round(price, 1),
            "logistics_score": round(logistics, 1),
            "timing_score": round(timing, 1),
            "regulatory_score": round(regulatory, 1),
            "reason": " | ".join(reason_parts),
            "reason_bullets": reason_parts,
            "nearest_port": nearest_port["name"] if nearest_port else "Unknown",
            "port_distance_km": nearest_port["distance_km"] if nearest_port else 0,
            "expected_export_window": export_window,
            "avg_price_usd_ton": stats.get("avg_price_usd_ton", 0),
            "confidence": round(confidence, 1),
            "cold_chain_required": stats.get("cold_chain_required", False),
            "processing_required": stats.get("processing_required", ""),
            "certification_days": stats.get("certification_days", 7),
        })

    results.sort(key=lambda x: x["export_score"], reverse=True)
    return results


def get_export_suggestion(
    crops: List[str],
    lat: float,
    lon: float,
    export_window: str = "next_quarter",
) -> Optional[Dict]:
    """
    Get the single best crop to export from the given location.

    Returns the top-scoring crop as a dict suitable for the API response,
    or None if no crops have export data.
    """
    scores = compute_export_scores(crops, lat, lon, export_window)

    if not scores or scores[0]["export_score"] == 0:
        return None

    top = scores[0]
    return {
        "crop": top["crop"],
        "export_score": top["export_score"],
        "reason": top["reason"],
        "reason_bullets": top.get("reason_bullets", []),
        "expected_export_window": top["expected_export_window"],
        "nearest_port": top["nearest_port"],
        "port_distance_km": top.get("port_distance_km", 0),
        "confidence": top["confidence"],
        "avg_price_usd_ton": top.get("avg_price_usd_ton", 0),
        "cold_chain_required": top.get("cold_chain_required", False),
        "processing_required": top.get("processing_required", ""),
    }
