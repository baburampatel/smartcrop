"""
Location resolver: converts PIN codes or lat,lon strings into structured
location data using offline lookups (pgeocode) — no external API required
for PIN codes.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

import pgeocode

logger = logging.getLogger(__name__)

# Initialize India postal code lookup (downloads ~2MB dataset on first run)
_nomi = pgeocode.Nominatim("IN")


@dataclass
class LocationInfo:
    """Structured location information."""
    latitude: float
    longitude: float
    state: Optional[str] = None
    district: Optional[str] = None
    pin_code: Optional[str] = None
    display_name: Optional[str] = None
    raw_input: str = ""


def _is_pin_code(location: str) -> bool:
    """Check if the location string looks like an Indian PIN code (6 digits)."""
    return bool(re.match(r"^\d{6}$", location.strip()))


def _is_coordinates(location: str) -> bool:
    """Check if the location string looks like lat,lon coordinates."""
    return bool(re.match(
        r"^-?\d+\.?\d*\s*,\s*-?\d+\.?\d*$", location.strip()
    ))


def _parse_coordinates(location: str) -> tuple:
    """Parse a 'lat,lon' string into (float, float)."""
    parts = location.strip().split(",")
    lat = float(parts[0].strip())
    lon = float(parts[1].strip())
    if not (-90 <= lat <= 90):
        raise ValueError(f"Latitude {lat} out of range [-90, 90]")
    if not (-180 <= lon <= 180):
        raise ValueError(f"Longitude {lon} out of range [-180, 180]")
    return lat, lon


def _geocode_pin(pin_code: str) -> LocationInfo:
    """Geocode an Indian PIN code using offline pgeocode database."""
    result = _nomi.query_postal_code(pin_code)

    # pgeocode returns NaN for unknown codes
    if result is None or (hasattr(result, 'latitude') and str(result.latitude) == 'nan'):
        raise ValueError(
            f"Could not geocode PIN code '{pin_code}'. "
            "Verify it is a valid Indian postal code."
        )

    lat = float(result.latitude)
    lon = float(result.longitude)
    state = str(result.state_name) if hasattr(result, 'state_name') and str(result.state_name) != 'nan' else None
    county = str(result.county_name) if hasattr(result, 'county_name') and str(result.county_name) != 'nan' else None
    place = str(result.place_name) if hasattr(result, 'place_name') and str(result.place_name) != 'nan' else None

    display = ", ".join(filter(None, [place, county, state, "India"]))

    return LocationInfo(
        latitude=lat,
        longitude=lon,
        state=state,
        district=county,
        pin_code=pin_code,
        display_name=display,
        raw_input=pin_code,
    )


def _reverse_lookup_nearest(lat: float, lon: float) -> dict:
    """Reverse-lookup: find nearest known location for coordinates."""
    rg = pgeocode.Nominatim("IN")
    # pgeocode doesn't have reverse geocoding, so we return empty
    # State/district will be unknown for coordinate inputs
    return {
        "state": None,
        "district": None,
        "display_name": f"{lat:.4f}, {lon:.4f}",
    }


def resolve_location(location: str) -> LocationInfo:
    """
    Resolve a location string to structured LocationInfo.

    Accepts:
        - Indian PIN code (e.g., '560001')
        - Coordinates as 'lat,lon' (e.g., '12.9716,77.5946')

    Returns:
        LocationInfo with lat, lon, state, district, etc.

    Raises:
        ValueError: If location format is unrecognized or geocoding fails.
    """
    location = location.strip()

    if not location:
        raise ValueError("Location string is empty")

    # Case 1: PIN code (offline lookup — always works)
    if _is_pin_code(location):
        logger.info("Resolving PIN code: %s (offline)", location)
        return _geocode_pin(location)

    # Case 2: Coordinates
    if _is_coordinates(location):
        logger.info("Parsing coordinates: %s", location)
        lat, lon = _parse_coordinates(location)
        geo = _reverse_lookup_nearest(lat, lon)

        return LocationInfo(
            latitude=lat,
            longitude=lon,
            state=geo.get("state"),
            district=geo.get("district"),
            display_name=geo.get("display_name", ""),
            raw_input=location,
        )

    raise ValueError(
        f"Unrecognized location format: '{location}'. "
        "Provide a 6-digit Indian PIN code or lat,lon coordinates."
    )
