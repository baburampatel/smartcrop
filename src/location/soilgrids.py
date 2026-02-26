"""
ISRIC SoilGrids v2.0 client: fetches modeled soil properties for a point.
Includes pedotransfer functions to estimate N/P/K from SOC, clay, and CEC.

API docs: https://rest.isric.org/soilgrids/v2.0/docs
License: CC-BY 4.0
"""

import logging
from typing import Optional, Dict

import requests

logger = logging.getLogger(__name__)

SOILGRIDS_BASE = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# Properties available in SoilGrids v2.0
SOILGRIDS_PROPERTIES = [
    "phh2o",   # pH in water (pH * 10)
    "clay",    # Clay content (g/kg)
    "sand",    # Sand content (g/kg)
    "silt",    # Silt content (g/kg)
    "soc",     # Soil organic carbon (dg/kg = g/kg * 10)
    "nitrogen", # Total nitrogen (cg/kg = g/kg * 100)
    "cec",     # Cation exchange capacity (mmol(c)/kg)
    "bdod",    # Bulk density (cg/cm3)
]

# Default depth interval for plough layer
DEFAULT_DEPTH = "0-5cm"


def fetch_soilgrids(
    lat: float,
    lon: float,
    depth: str = DEFAULT_DEPTH,
    properties: list = None,
) -> Dict[str, Optional[float]]:
    """
    Fetch soil properties from ISRIC SoilGrids for a given point.

    Args:
        lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180)
        depth: Depth interval (default '0-5cm')
        properties: List of SoilGrids property names (default: all)

    Returns:
        Dict with canonical soil property names and values.
        Values are None if the API is unavailable or the property is missing.
    """
    if properties is None:
        properties = SOILGRIDS_PROPERTIES

    params = {
        "lat": lat,
        "lon": lon,
        "depth": depth,
        "value": "mean",
    }
    # Add each property as a separate query param
    param_list = [(k, v) for k, v in params.items()]
    for prop in properties:
        param_list.append(("property", prop))

    result = {
        "pH": None,
        "clay_pct": None,
        "sand_pct": None,
        "silt_pct": None,
        "soc_g_kg": None,
        "nitrogen_g_kg": None,
        "cec_mmol_kg": None,
        "bulk_density_kg_m3": None,
    }

    try:
        resp = requests.get(
            SOILGRIDS_BASE, params=param_list, timeout=30,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        logger.warning("SoilGrids API request failed: %s", e)
        return result

    # Parse layers from response
    layers = data.get("properties", {}).get("layers", [])
    for layer in layers:
        name = layer.get("name", "")
        depths = layer.get("depths", [])
        if not depths:
            continue
        # Find matching depth
        value = None
        for d in depths:
            label = d.get("label", "")
            if depth.replace("cm", "") in label or label == depth:
                values_dict = d.get("values", {})
                value = values_dict.get("mean")
                break
        # Fallback: use first depth
        if value is None and depths:
            value = depths[0].get("values", {}).get("mean")

        if value is not None:
            if name == "phh2o":
                result["pH"] = value / 10.0  # SoilGrids stores pH * 10
            elif name == "clay":
                result["clay_pct"] = value / 10.0  # g/kg -> %
            elif name == "sand":
                result["sand_pct"] = value / 10.0
            elif name == "silt":
                result["silt_pct"] = value / 10.0
            elif name == "soc":
                result["soc_g_kg"] = value / 10.0  # dg/kg -> g/kg
            elif name == "nitrogen":
                result["nitrogen_g_kg"] = value / 100.0  # cg/kg -> g/kg
            elif name == "cec":
                result["cec_mmol_kg"] = float(value)
            elif name == "bdod":
                result["bulk_density_kg_m3"] = value * 10.0  # cg/cm3 -> kg/m3

    return result


def estimate_npk_from_soil(soil: Dict[str, Optional[float]]) -> Dict[str, float]:
    """
    Estimate plant-available N, P, K (kg/ha) from soil properties using
    pedotransfer functions.

    These are rough agronomic approximations used when lab data is unavailable.

    Args:
        soil: Dict from fetch_soilgrids() with SOC, CEC, clay, etc.

    Returns:
        Dict with estimated N_kg_ha, P_kg_ha, K_kg_ha values.
    """
    # Default values if estimation fails
    n_kg_ha = 50.0
    p_kg_ha = 25.0
    k_kg_ha = 25.0

    soc = soil.get("soc_g_kg")
    nitrogen = soil.get("nitrogen_g_kg")
    cec = soil.get("cec_mmol_kg")
    clay = soil.get("clay_pct")

    # Estimate N from total nitrogen
    # Typical available N is ~2-5% of total N in top 15cm
    # Bulk density ~1300 kg/m3, depth 0.15m -> soil mass ~1950 t/ha
    if nitrogen is not None:
        # nitrogen in g/kg * 1950 t/ha * fraction available (~0.03)
        n_kg_ha = nitrogen * 1950 * 0.03
        n_kg_ha = max(5.0, min(n_kg_ha, 400.0))  # clamp
    elif soc is not None:
        # Fallback: N ~ SOC / 10 (C:N ratio ~10)
        total_n_g_kg = soc / 10.0
        n_kg_ha = total_n_g_kg * 1950 * 0.03
        n_kg_ha = max(5.0, min(n_kg_ha, 400.0))

    # Estimate P from SOC and clay (Bray P approximation)
    if soc is not None:
        # Available P correlates loosely with organic matter
        p_kg_ha = soc * 2.5 + 5.0
        p_kg_ha = max(3.0, min(p_kg_ha, 200.0))

    # Estimate K from CEC and clay
    if cec is not None:
        # K typically 2-5% of CEC in meq/100g
        # CEC in mmol(c)/kg = meq/kg, divide by 10 for meq/100g
        cec_meq_100g = cec / 10.0
        k_kg_ha = cec_meq_100g * 0.03 * 39.1 * 20  # K+ atomic weight, soil mass
        k_kg_ha = max(5.0, min(k_kg_ha, 400.0))
    elif clay is not None:
        # Clay-rich soils tend to have more exchangeable K
        k_kg_ha = clay * 1.5 + 10.0
        k_kg_ha = max(5.0, min(k_kg_ha, 400.0))

    return {
        "N_kg_ha": round(n_kg_ha, 1),
        "P_kg_ha": round(p_kg_ha, 1),
        "K_kg_ha": round(k_kg_ha, 1),
    }
