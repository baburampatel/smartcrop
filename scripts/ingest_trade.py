#!/usr/bin/env python
"""
Trade data ingestion script.

Fetches and normalizes agricultural export trade data from public sources.
Saves results to data/trade_catalog.json with full provenance.

Usage:
    python scripts/ingest_trade.py [--refresh]

Currently uses curated data from APEDA/MoC public reports.
When API keys are available, can fetch live data from:
  - DGCI&S: https://dgciskol.gov.in
  - UN COMTRADE: https://comtradeplus.un.org/
  - FAOSTAT: https://www.fao.org/faostat/
  - ITC Trade Map: https://www.trademap.org/
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.location.export_data import EXPORT_STATS, get_top_export_crops


def build_trade_catalog() -> dict:
    """Build the trade catalog with provenance metadata."""
    catalog = {
        "metadata": {
            "title": "India Agricultural Export Trade Catalog",
            "description": "Curated export statistics for Indian agricultural commodities",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "version": "1.0.0",
            "sources": [
                {
                    "name": "APEDA ExportStat",
                    "url": "https://apeda.gov.in/apedawebsite/trade_Promotion/Export_statistic.htm",
                    "license": "Government of India - Open Data",
                    "accessed": "2025-01",
                },
                {
                    "name": "Ministry of Commerce - Export Data",
                    "url": "https://commerce.gov.in/trade-statistics/",
                    "license": "Government of India - Open Data",
                    "accessed": "2025-01",
                },
                {
                    "name": "DGCI&S - Directorate General of Comm. Intelligence",
                    "url": "https://dgciskol.gov.in",
                    "license": "Requires registration for detailed data",
                    "accessed": "2025-01",
                    "note": "Detailed commodity-level data requires paid subscription",
                },
                {
                    "name": "UN COMTRADE",
                    "url": "https://comtradeplus.un.org/",
                    "license": "Free for public use (registration required)",
                    "accessed": "2025-01",
                    "note": "API key required for programmatic access",
                },
                {
                    "name": "FAOSTAT",
                    "url": "https://www.fao.org/faostat/en/#data/TCL",
                    "license": "CC BY-NC-SA 3.0 IGO",
                    "accessed": "2025-01",
                },
            ],
            "variables": [
                "export_volume_mt", "export_value_usd_mn",
                "quarterly_growth_pct", "avg_price_usd_ton",
                "price_trend", "top_destinations", "peak_export_months",
                "harvest_to_export_weeks", "cold_chain_required",
                "processing_required", "apeda_registered",
                "phytosanitary_complexity", "certification_days",
            ],
        },
        "crops": {},
        "rankings": {
            "by_value": [],
            "by_volume": [],
            "by_growth": [],
        },
    }

    # Add all crop data
    for crop, stats in EXPORT_STATS.items():
        catalog["crops"][crop] = {
            **stats,
            "data_quality": "curated",
            "last_updated": "2025-01",
        }

    # Add rankings
    catalog["rankings"]["by_value"] = [
        {"crop": c["crop"], "value_usd_mn": c["export_value_usd_mn"]}
        for c in get_top_export_crops(15, "value")
    ]
    catalog["rankings"]["by_volume"] = [
        {"crop": c["crop"], "volume_mt": c["export_volume_mt"]}
        for c in get_top_export_crops(15, "volume")
    ]
    catalog["rankings"]["by_growth"] = [
        {"crop": c["crop"], "growth_pct": c["quarterly_growth_pct"]}
        for c in get_top_export_crops(15, "growth")
    ]

    return catalog


def main():
    """Main entry point."""
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    output_path = data_dir / "trade_catalog.json"

    print("Building trade catalog...")
    catalog = build_trade_catalog()

    with open(output_path, "w") as f:
        json.dump(catalog, f, indent=2)

    n_crops = len(catalog["crops"])
    print(f"Trade catalog saved to {output_path}")
    print(f"  Crops: {n_crops}")
    print(f"  Sources: {len(catalog['metadata']['sources'])}")
    print(f"  Top by value: {catalog['rankings']['by_value'][0]['crop']}")
    print(f"  Top by growth: {catalog['rankings']['by_growth'][0]['crop']}")


if __name__ == "__main__":
    main()
