"""
Export scorer: computes an export_score (0–100) for each candidate crop
and selects the best crop to export from a given location.

FIXED ALGORITHM (v2):
  Final Score = 0.35 * demand + 0.20 * price + 0.20 * logistics + 0.15 * timing + 0.10 * regulatory

  Key improvements over v1:
  - State-level agroclimatic suitability weights: crops not suited to a state get heavy penalty.
  - ML-recommendation boost: crops the model recommends for that soil get +15 pts.
  - Seasonal availability filter: crops out of season penalized.
  - Jitter-free: deterministic tie-breaking by crop name ensures consistent results.
"""

import datetime
import logging
from typing import Dict, List, Optional

from src.location.export_data import get_export_stats
from src.location.trade_routes import find_nearest_ports, get_logistics_score

logger = logging.getLogger(__name__)

# ── Score weights ──────────────────────────────────────────────────────────────
W_DEMAND    = 0.35
W_PRICE     = 0.20
W_LOGISTICS = 0.20
W_TIMING    = 0.15
W_REGULATORY = 0.10

# ── State-level agroclimatic suitability ──────────────────────────────────────
# Score 0–100: how well can this crop be commercially grown in this state?
# Sources: ICAR agroclimatic zone maps, state agriculture dept crop calendars.
# 100 = primary producer; 60 = can grow; 20 = marginal; 0 = not grown.
STATE_CROP_SUITABILITY: Dict[str, Dict[str, int]] = {
    # --- South India ---
    "andhra pradesh": {
        "rice": 90, "groundnut": 85, "cotton": 80, "maize": 75,
        "sugarcane": 70, "chilli": 60, "tobacco": 60, "soybean": 40,
        "wheat": 20, "mustard": 20, "coffee": 10, "banana": 65,
        "coconut": 70, "potato": 30, "chickpea": 50, "lentil": 30,
    },
    "telangana": {
        "rice": 85, "cotton": 85, "maize": 80, "soybean": 75,
        "groundnut": 70, "sugarcane": 65, "mustard": 20,
        "wheat": 20, "coffee": 10, "banana": 60, "coconut": 40,
    },
    "karnataka": {
        "rice": 80, "sugarcane": 85, "maize": 80, "groundnut": 75,
        "cotton": 70, "coffee": 95, "coconut": 85, "banana": 75,
        "ragi": 80, "wheat": 30, "mustard": 20, "soybean": 60,
        "chickpea": 50, "lentil": 30,
    },
    "kerala": {
        "coconut": 95, "banana": 85, "coffee": 80, "rice": 70,
        "sugarcane": 60, "groundnut": 30, "cotton": 20, "wheat": 10,
        "soybean": 20, "potato": 40,
    },
    "tamil nadu": {
        "rice": 88, "sugarcane": 85, "banana": 80, "groundnut": 75,
        "cotton": 65, "coconut": 85, "maize": 60, "coffee": 40,
        "wheat": 20, "soybean": 30, "potato": 35,
    },
    # --- West India ---
    "maharashtra": {
        "sugarcane": 90, "cotton": 85, "soybean": 85, "groundnut": 70,
        "maize": 70, "rice": 65, "wheat": 60, "onion": 85,
        "banana": 70, "potato": 50, "chickpea": 60, "mustard": 40,
    },
    "gujarat": {
        "groundnut": 90, "cotton": 88, "mustard": 75, "wheat": 70,
        "sugarcane": 70, "castor": 80, "maize": 60, "rice": 55,
        "soybean": 50, "potato": 55, "chickpea": 60,
    },
    "rajasthan": {
        "mustard": 90, "wheat": 85, "groundnut": 70, "maize": 60,
        "cotton": 55, "chickpea": 75, "sugarcane": 30, "rice": 40,
        "soybean": 40, "potato": 50, "lentil": 65,
    },
    # --- North India ---
    "punjab": {
        "wheat": 95, "rice": 88, "maize": 70, "potato": 75,
        "sugarcane": 65, "cotton": 60, "mustard": 55,
        "groundnut": 20, "soybean": 25, "chickpea": 50, "lentil": 50,
    },
    "haryana": {
        "wheat": 92, "rice": 80, "mustard": 80, "sugarcane": 70,
        "cotton": 65, "maize": 60, "potato": 65, "groundnut": 25,
        "soybean": 30, "chickpea": 55, "lentil": 55,
    },
    "uttar pradesh": {
        "wheat": 90, "sugarcane": 88, "rice": 80, "potato": 80,
        "mustard": 75, "maize": 65, "lentil": 70, "chickpea": 70,
        "groundnut": 40, "cotton": 35, "soybean": 50,
    },
    "madhya pradesh": {
        "soybean": 90, "wheat": 85, "chickpea": 85, "mustard": 75,
        "maize": 70, "rice": 65, "cotton": 60, "groundnut": 55,
        "lentil": 75, "sugarcane": 50, "potato": 55,
    },
    # --- East India ---
    "west bengal": {
        "rice": 92, "jute": 90, "potato": 85, "maize": 60,
        "mustard": 75, "wheat": 65, "sugarcane": 60, "banana": 65,
        "coconut": 50, "groundnut": 35, "soybean": 30,
    },
    "bihar": {
        "rice": 85, "wheat": 80, "maize": 75, "lentil": 80,
        "mustard": 70, "sugarcane": 70, "potato": 75,
        "groundnut": 40, "soybean": 35, "cotton": 20,
    },
    "odisha": {
        "rice": 90, "groundnut": 65, "maize": 65, "sugarcane": 60,
        "potato": 55, "jute": 60, "coconut": 65,
        "wheat": 35, "soybean": 40, "mustard": 40,
    },
    # --- Northeast ---
    "assam": {
        "rice": 88, "jute": 80, "tea": 90, "banana": 70,
        "potato": 50, "mustard": 60, "wheat": 40, "maize": 55,
    },
    # --- Other states (default moderate) ---
}

# Default suitability for unknown states
DEFAULT_SUITABILITY: Dict[str, int] = {
    "rice": 60, "wheat": 50, "maize": 50, "soybean": 50,
    "groundnut": 50, "sugarcane": 50, "cotton": 50, "coffee": 30,
    "banana": 40, "coconut": 40, "chickpea": 50, "mustard": 50,
    "potato": 40, "jute": 30, "lentil": 50,
}


def _get_suitability(crop: str, state: Optional[str]) -> int:
    """Get agroclimatic suitability score (0–100) for crop in state."""
    if not state:
        return DEFAULT_SUITABILITY.get(crop.lower(), 50)
    state_lower = state.lower().strip()
    suitability_map = STATE_CROP_SUITABILITY.get(state_lower, DEFAULT_SUITABILITY)
    return suitability_map.get(crop.lower(), 45)  # 45 = unknown crop, not penalty


def _demand_score(stats: Dict) -> float:
    """Score 0–100 based on quarterly growth percentage."""
    growth = stats.get("quarterly_growth_pct", 0)
    # Map: -20% → 5, 0% → 40, +20% → 80, +30% → 100
    score = 40 + growth * 2
    return max(5, min(100, score))


def _price_score(stats: Dict) -> float:
    """Score 0–100 based on price trend."""
    trend = stats.get("price_trend", "stable")
    return {"rising": 85, "stable": 50, "falling": 15}.get(trend, 50)


def _timing_score(stats: Dict, target_month: int) -> float:
    """Score 0–100 based on whether harvest aligns with export window."""
    peak = stats.get("peak_export_months", [])
    if not peak:
        return 40

    if target_month in peak:
        return 90
    # Adjacent months
    adj = [(target_month - 1) % 12 + 1, target_month % 12 + 1]
    if any(m in peak for m in adj):
        return 60
    # 2 months away
    adj2 = [(target_month - 2) % 12 + 1, (target_month + 1) % 12 + 1]
    if any(m in peak for m in adj2):
        return 35
    return 15


def _regulatory_score(stats: Dict) -> float:
    """Score 0–100 based on certification complexity."""
    complexity = stats.get("phytosanitary_complexity", "medium")
    return {"low": 90, "medium": 60, "high": 30}.get(complexity, 60)


def compute_export_scores(
    crops: List[str],
    lat: float,
    lon: float,
    state: Optional[str] = None,
    model_recommended: Optional[List[str]] = None,
    export_window: str = "next_quarter",
) -> List[Dict]:
    """
    Compute export scores for candidate crops at a location.

    Args:
        crops: Candidate crop names to evaluate.
        lat: Latitude.
        lon: Longitude.
        state: Indian state name (used for suitability weighting).
        model_recommended: Crops the ML model recommended for this location (get +15pt boost).
        export_window: Timing target.

    Returns:
        List of scored crops sorted by export_score descending.
    """
    # Determine target month for export window
    now = datetime.datetime.now()
    if export_window == "next_quarter":
        current_quarter = (now.month - 1) // 3 + 1
        target_month = (current_quarter * 3) % 12 + 1
    else:
        target_month = now.month

    # Nearest port for logistics
    nearest_ports = find_nearest_ports(lat, lon, n=1)
    nearest_port = nearest_ports[0] if nearest_ports else None

    ml_recommended_set = {c.lower() for c in (model_recommended or [])}

    # Deduplicate crops (case-insensitive)
    seen = set()
    unique_crops = []
    for c in crops:
        cl = c.lower().strip()
        if cl not in seen:
            seen.add(cl)
            unique_crops.append(cl)

    results = []
    for crop in unique_crops:
        stats = get_export_stats(crop)
        if stats is None:
            continue  # Skip crops with no export data

        # ── Component scores ──
        demand    = _demand_score(stats)
        price     = _price_score(stats)
        logistics_info = get_logistics_score(lat, lon, stats.get("cold_chain_required", False))
        logistics = logistics_info["score"]
        timing    = _timing_score(stats, target_month)
        regulatory = _regulatory_score(stats)

        # ── Agroclimatic suitability multiplier ──
        suitability = _get_suitability(crop, state)    # 0–100
        # Penalty: if suitability < 40, heavily penalize (crop can't grow here)
        if suitability < 40:
            suitability_multiplier = 0.3
        elif suitability < 60:
            suitability_multiplier = 0.7
        else:
            suitability_multiplier = 1.0

        # ── Weighted base score ──
        base = (
            W_DEMAND    * demand +
            W_PRICE     * price +
            W_LOGISTICS * logistics +
            W_TIMING    * timing +
            W_REGULATORY * regulatory
        )

        # Apply suitability multiplier
        total = base * suitability_multiplier

        # ── ML recommendation boost: +15 if this crop suits this soil ──
        if crop in ml_recommended_set:
            total = min(100, total + 15)

        # ── Build reason bullets ──
        reason_parts = []
        growth = stats.get("quarterly_growth_pct", 0)
        if growth > 5:
            reason_parts.append(f"Strong export demand (+{growth}% q/q)")
        elif growth > 0:
            reason_parts.append(f"Positive export demand (+{growth}% q/q)")
        else:
            reason_parts.append(f"Moderate export demand ({growth}% q/q)")

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

        if crop in ml_recommended_set:
            reason_parts.append("Soil & climate suitable for this location (AI-verified)")

        if suitability >= 70:
            reason_parts.append(f"Agroclimatic suitability: High ({suitability}/100)")
        elif suitability >= 50:
            reason_parts.append(f"Agroclimatic suitability: Moderate ({suitability}/100)")

        confidence = min(92, total * 0.92)

        results.append({
            "crop": crop,
            "export_score": round(total, 1),
            "demand_score": round(demand, 1),
            "price_score": round(price, 1),
            "logistics_score": round(logistics, 1),
            "timing_score": round(timing, 1),
            "regulatory_score": round(regulatory, 1),
            "suitability_score": suitability,
            "ml_recommended": crop in ml_recommended_set,
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

    # Sort: by export_score desc; tie-break by crop name for stability
    results.sort(key=lambda x: (-x["export_score"], x["crop"]))
    return results


def get_export_suggestion(
    crops: List[str],
    lat: float,
    lon: float,
    state: Optional[str] = None,
    model_recommended: Optional[List[str]] = None,
    export_window: str = "next_quarter",
) -> Optional[Dict]:
    """
    Get the single best crop to export from the given location.

    Returns the top-scoring crop, or None if no crops have export data.
    """
    scores = compute_export_scores(
        crops, lat, lon,
        state=state,
        model_recommended=model_recommended,
        export_window=export_window,
    )

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
