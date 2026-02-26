"""
CLI entry point for the location pipeline.

Usage:
    python scripts/location_pipeline.py --location 560001 --pipeline Tertiary
    python scripts/location_pipeline.py --location 12.9716,77.5946 --pipeline Tertiary --mode dry_run
    python scripts/location_pipeline.py --location 560001 --pipeline Primary --target-crops rice,maize,wheat
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.location.pipeline import LocationPipeline, PipelineConfig


def main():
    parser = argparse.ArgumentParser(
        description="Run location-based crop recommendation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/location_pipeline.py --location 560001
  python scripts/location_pipeline.py --location 12.9716,77.5946 --pipeline Tertiary
  python scripts/location_pipeline.py --location 560001 --target-crops rice,maize,wheat
        """,
    )
    parser.add_argument(
        "--location", required=True,
        help="Indian PIN code (e.g., 560001) or lat,lon (e.g., 12.97,77.59)",
    )
    parser.add_argument(
        "--pipeline", default="Tertiary",
        choices=["Primary", "Secondary", "Tertiary"],
        help="Pipeline tier (default: Tertiary)",
    )
    parser.add_argument(
        "--target-crops", default="rice,maize,wheat,soybean,groundnut,sugarcane",
        help="Comma-separated target crops",
    )
    parser.add_argument(
        "--prioritize", default="data_quality",
        choices=["data_quality", "cost", "speed"],
        help="Optimization priority (default: data_quality)",
    )
    parser.add_argument(
        "--radius-km", type=int, default=50,
        help="Search radius in km (default: 50)",
    )
    parser.add_argument(
        "--mode", default="full_run",
        choices=["full_run", "data_only", "dry_run"],
        help="Run mode (default: full_run)",
    )
    parser.add_argument(
        "--lab-data", default=None,
        help="JSON string with lab soil data (for Primary pipeline)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Parse lab data if provided
    lab_data = None
    if args.lab_data:
        try:
            lab_data = json.loads(args.lab_data)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON for --lab-data: {e}", file=sys.stderr)
            sys.exit(1)

    # Build config
    config = PipelineConfig(
        location=args.location,
        pipeline=args.pipeline,
        target_crops=args.target_crops.split(","),
        prioritize=args.prioritize,
        radius_km=args.radius_km,
        mode=args.mode,
        lab_data=lab_data,
    )

    # Dry run: just resolve location and report
    if args.mode == "dry_run":
        from src.location.resolver import resolve_location
        loc = resolve_location(args.location)
        result = {
            "mode": "dry_run",
            "location_resolved": {
                "latitude": loc.latitude,
                "longitude": loc.longitude,
                "state": loc.state,
                "district": loc.district,
                "pin_code": loc.pin_code,
            },
            "pipeline": args.pipeline,
            "target_crops": config.target_crops,
            "message": "Dry run complete. Location resolved successfully.",
        }
        print(json.dumps(result, indent=2))
        return

    # Run pipeline
    pipeline = LocationPipeline()
    result = pipeline.run(config)

    # Output JSON
    output = {
        "location_info": result.location_info,
        "soil_data": result.soil_data,
        "weather_data": result.weather_data,
        "market_data": result.market_data,
        "model_input": result.model_input,
        "data_sources": result.data_sources,
        "pipeline_used": result.pipeline_used,
        "warnings": result.warnings,
    }

    if args.mode == "data_only":
        output["message"] = (
            "Data acquisition complete. Use model_input with /predict endpoint."
        )
        print(json.dumps(output, indent=2))
        return

    # Full run: also call the model (if available)
    print("\n--- Pipeline Data Acquisition Result ---")
    print(json.dumps(output, indent=2))

    print("\n--- Model Input (ready for /predict endpoint) ---")
    print(json.dumps(result.model_input, indent=2))

    print("\n--- Execution Prompt ---")
    crops_str = ",".join(config.target_crops)
    print(
        f"RUN_LOCATION_PIPELINE LOCATION={args.location} "
        f"PIPELINE={args.pipeline} TARGET_CROPS={crops_str} "
        f"PRIORITIZE={args.prioritize} RADIUS_KM={args.radius_km} "
        f"MODE=full_run"
    )


if __name__ == "__main__":
    main()
