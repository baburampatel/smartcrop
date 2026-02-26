"""
Trade routes & logistics: nearest port lookup, transit estimates,
and phytosanitary/certification requirements for Indian agri-exports.

Port data from Indian Ports Association (IPA) and APEDA.
"""

import math
from typing import Dict, List, Optional

# ---- Major Indian Agricultural Export Ports ----
# Coordinates from public sources; capacity = annual agri-cargo throughput (approx MT)
EXPORT_PORTS = [
    {
        "name": "JNPT (Nhava Sheva)",
        "city": "Mumbai",
        "state": "Maharashtra",
        "lat": 18.9490,
        "lon": 72.9510,
        "type": "container",
        "agri_capacity_mt": 5_000_000,
        "cold_storage": True,
        "fumigation": True,
    },
    {
        "name": "Chennai Port",
        "city": "Chennai",
        "state": "Tamil Nadu",
        "lat": 13.0827,
        "lon": 80.2707,
        "type": "container",
        "agri_capacity_mt": 3_000_000,
        "cold_storage": True,
        "fumigation": True,
    },
    {
        "name": "Mundra Port",
        "city": "Mundra",
        "state": "Gujarat",
        "lat": 22.8394,
        "lon": 69.7250,
        "type": "container",
        "agri_capacity_mt": 4_500_000,
        "cold_storage": True,
        "fumigation": True,
    },
    {
        "name": "Kochi Port",
        "city": "Kochi",
        "state": "Kerala",
        "lat": 9.9312,
        "lon": 76.2673,
        "type": "container",
        "agri_capacity_mt": 1_500_000,
        "cold_storage": True,
        "fumigation": True,
    },
    {
        "name": "Visakhapatnam Port",
        "city": "Visakhapatnam",
        "state": "Andhra Pradesh",
        "lat": 17.6868,
        "lon": 83.2185,
        "type": "bulk",
        "agri_capacity_mt": 2_000_000,
        "cold_storage": False,
        "fumigation": True,
    },
    {
        "name": "Kolkata Port (KoPT)",
        "city": "Kolkata",
        "state": "West Bengal",
        "lat": 22.5726,
        "lon": 88.3639,
        "type": "container",
        "agri_capacity_mt": 1_800_000,
        "cold_storage": True,
        "fumigation": True,
    },
    {
        "name": "Tuticorin Port",
        "city": "Thoothukudi",
        "state": "Tamil Nadu",
        "lat": 8.7642,
        "lon": 78.1348,
        "type": "bulk",
        "agri_capacity_mt": 1_200_000,
        "cold_storage": False,
        "fumigation": True,
    },
    {
        "name": "Kandla Port",
        "city": "Gandhidham",
        "state": "Gujarat",
        "lat": 23.0225,
        "lon": 70.2167,
        "type": "bulk",
        "agri_capacity_mt": 3_000_000,
        "cold_storage": False,
        "fumigation": True,
    },
    {
        "name": "Mangalore Port (NMPT)",
        "city": "Mangalore",
        "state": "Karnataka",
        "lat": 12.9141,
        "lon": 74.8560,
        "type": "bulk",
        "agri_capacity_mt": 800_000,
        "cold_storage": False,
        "fumigation": True,
    },
    {
        "name": "Kakinada Port",
        "city": "Kakinada",
        "state": "Andhra Pradesh",
        "lat": 16.9891,
        "lon": 82.2475,
        "type": "bulk",
        "agri_capacity_mt": 1_000_000,
        "cold_storage": False,
        "fumigation": True,
    },
]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in km using Haversine formula."""
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def find_nearest_ports(lat: float, lon: float, n: int = 3) -> List[Dict]:
    """
    Find the N nearest export ports to a given location.

    Returns list of ports with distance and estimated transit time.
    Assumes average road speed of 40 km/h for agri-cargo in India.
    """
    ROAD_SPEED_KMH = 40  # avg truck speed for agri-cargo

    ports_with_dist = []
    for port in EXPORT_PORTS:
        dist = _haversine_km(lat, lon, port["lat"], port["lon"])
        # Road distance is roughly 1.3x straight-line in India
        road_dist = dist * 1.3
        transit_hours = road_dist / ROAD_SPEED_KMH

        ports_with_dist.append({
            "name": port["name"],
            "city": port["city"],
            "state": port["state"],
            "distance_km": round(road_dist, 0),
            "est_transit_hours": round(transit_hours, 1),
            "cold_storage": port["cold_storage"],
            "fumigation": port["fumigation"],
            "type": port["type"],
        })

    ports_with_dist.sort(key=lambda x: x["distance_km"])
    return ports_with_dist[:n]


def get_logistics_score(lat: float, lon: float, cold_chain_needed: bool) -> Dict:
    """
    Compute a logistics feasibility score (0–100) for exporting from location.

    Factors:
    - Distance to nearest port (closer = better)
    - Cold storage availability at port (if needed)
    - Port type (container = easier for processed goods)
    """
    nearest = find_nearest_ports(lat, lon, n=1)[0]
    dist = nearest["distance_km"]

    # Distance score: 100 at 0km, 0 at 2000km
    dist_score = max(0, 100 - (dist / 20))

    # Cold chain penalty
    cold_penalty = 0
    if cold_chain_needed and not nearest["cold_storage"]:
        cold_penalty = -25

    # Container port bonus
    port_bonus = 10 if nearest["type"] == "container" else 0

    score = max(0, min(100, dist_score + cold_penalty + port_bonus))

    return {
        "score": round(score, 1),
        "nearest_port": nearest["name"],
        "distance_km": nearest["distance_km"],
        "transit_hours": nearest["est_transit_hours"],
        "cold_storage_available": nearest["cold_storage"],
    }


# ---- APEDA Certification & Phytosanitary Rules ----
CERTIFICATION_STEPS = {
    "low": [
        "Register as APEDA exporter (one-time)",
        "Obtain phytosanitary certificate from Plant Quarantine Office",
        "Self-declaration of compliance",
    ],
    "medium": [
        "Register as APEDA exporter (one-time)",
        "Get commodity tested at NABL-accredited lab",
        "Obtain phytosanitary certificate",
        "Aflatoxin / pesticide residue certificate",
    ],
    "high": [
        "Register as APEDA exporter (one-time)",
        "Pre-shipment NABL lab testing (MRL compliance)",
        "Cold chain documentation and temperature log",
        "Obtain phytosanitary + fumigation certificate",
        "Destination-country import permit verification",
    ],
}


def get_certification_info(complexity: str) -> Dict:
    """Get certification steps and estimated time for given complexity level."""
    steps = CERTIFICATION_STEPS.get(complexity, CERTIFICATION_STEPS["low"])
    days_map = {"low": 7, "medium": 14, "high": 21}
    return {
        "complexity": complexity,
        "steps": steps,
        "estimated_days": days_map.get(complexity, 7),
    }
