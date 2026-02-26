"""
One-command crop recommendation: type a location, get results.

Usage:
    python scripts/recommend.py
    python scripts/recommend.py --location 560001
    python scripts/recommend.py --location 12.9716,77.5946
"""

import json
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)


def print_banner():
    print("=" * 60)
    print("   CROP RECOMMENDATION SYSTEM")
    print("   Type a location -> Get top-3 crop recommendations")
    print("=" * 60)
    print()


def load_model():
    """Load the trained model. Returns (classifier, yield_model, pipeline, explainer)."""
    from src.models.classifier import CropClassifier
    from src.models.yield_model import YieldPredictor
    from src.data.features import load_pipeline
    from src.explain.shap_explain import SHAPExplainer

    models_dir = PROJECT_ROOT / "models"

    clf_path = models_dir / "crop_classifier.joblib"
    if not clf_path.exists():
        print("[!] Model not found. Training first...")
        print("    Running: python scripts/ingest.py --data-dir data")
        import subprocess
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "ingest.py"),
                        "--data-dir", str(PROJECT_ROOT / "data")], check=True)
        print("    Running: python scripts/train.py --data-dir data")
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "train.py"),
                        "--data-dir", str(PROJECT_ROOT / "data"),
                        "--model-type", "xgboost"], check=True)

    classifier = CropClassifier.load(str(clf_path))

    yield_model = None
    yield_path = models_dir / "yield_predictor.joblib"
    if yield_path.exists():
        yield_model = YieldPredictor.load(str(yield_path))

    feature_pipeline = None
    pipe_path = models_dir / "feature_pipeline.joblib"
    if pipe_path.exists():
        feature_pipeline = load_pipeline(str(pipe_path))

    explainer = None
    try:
        explainer = SHAPExplainer(classifier.model, classifier.feature_names)
    except Exception:
        pass

    return classifier, yield_model, feature_pipeline, explainer


def fetch_location_data(location: str) -> dict:
    """Fetch soil + weather data for a location."""
    from src.location.pipeline import LocationPipeline, PipelineConfig

    config = PipelineConfig(location=location, pipeline="Tertiary")
    pipeline = LocationPipeline()
    result = pipeline.run(config)
    return result


def predict(model_input: dict, classifier, yield_model, feature_pipeline, explainer):
    """Run prediction and return results."""
    import pandas as pd
    from src.data.features import get_model_features

    input_data = {
        "N_kg_ha": model_input["N"],
        "P_kg_ha": model_input["P"],
        "K_kg_ha": model_input["K"],
        "avg_temp_c": model_input["temperature"],
        "humidity_pct": model_input["humidity"],
        "pH": model_input["ph"],
        "avg_precip_mm": model_input["rainfall"],
    }
    df = pd.DataFrame([input_data])

    if feature_pipeline is not None:
        df = feature_pipeline.transform(df)

    for col in classifier.feature_names:
        if col not in df.columns:
            df[col] = 0

    X = df[classifier.feature_names].astype(float).fillna(0).values
    top_k = classifier.predict_top_k(X, k=3)

    results = []
    for pred in top_k[0]:
        crop = pred["crop"]
        prob = pred["probability"]

        yld = 0.0
        if yield_model is not None:
            yld = yield_model.predict_for_crop(X, crop)

        explanation = []
        if explainer is not None:
            try:
                explanation = explainer.explain_prediction(X, pred["class_idx"], top_features=3)
            except Exception:
                pass

        results.append({
            "crop": crop,
            "probability": round(prob * 100, 1),
            "yield_kg_ha": round(yld, 0),
            "explanation": explanation,
        })

    return results


def print_results(location_info, soil_data, weather_data, predictions, data_sources):
    """Pretty-print the results."""
    print()
    print("-" * 60)
    print("  LOCATION")
    print("-" * 60)
    lat = location_info.get("latitude", "?")
    lon = location_info.get("longitude", "?")
    state = location_info.get("state") or "Unknown"
    district = location_info.get("district") or "Unknown"
    print(f"  Coordinates : {lat}, {lon}")
    print(f"  State       : {state}")
    print(f"  District    : {district}")

    print()
    print("-" * 60)
    print("  SOIL DATA")
    print("-" * 60)
    print(f"  pH          : {soil_data.get('pH', '?')}")
    print(f"  Nitrogen    : {soil_data.get('N_kg_ha', '?')} kg/ha")
    print(f"  Phosphorus  : {soil_data.get('P_kg_ha', '?')} kg/ha")
    print(f"  Potassium   : {soil_data.get('K_kg_ha', '?')} kg/ha")
    src = soil_data.get("source", "unknown")
    print(f"  Source      : {src}")

    print()
    print("-" * 60)
    print("  WEATHER DATA")
    print("-" * 60)
    print(f"  Avg Temp    : {weather_data.get('avg_temp_c', '?')} C")
    print(f"  Humidity    : {weather_data.get('humidity_pct', '?')} %")
    print(f"  Rainfall    : {weather_data.get('avg_precip_mm', '?')} mm/year")

    print()
    print("=" * 60)
    print("  TOP-3 CROP RECOMMENDATIONS")
    print("=" * 60)
    for i, pred in enumerate(predictions, 1):
        crop = pred["crop"].upper()
        prob = pred["probability"]
        yld = pred["yield_kg_ha"]
        print()
        print(f"  #{i}  {crop}")
        print(f"      Confidence : {prob}%")
        if yld > 0:
            print(f"      Est. Yield : {yld:,.0f} kg/ha")
        if pred.get("explanation"):
            factors = ", ".join(
                f"{e['feature']}({e['contribution']:+.2f})"
                for e in pred["explanation"][:3]
            )
            print(f"      Key factors: {factors}")

    print()
    print("-" * 60)
    print("  Data sources:", " | ".join(data_sources))
    print("-" * 60)
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Crop Recommendation by Location")
    parser.add_argument("--location", "-l", default=None,
                        help="PIN code or lat,lon (interactive if omitted)")
    args = parser.parse_args()

    print_banner()

    # Load model
    print("[*] Loading model...")
    classifier, yield_model, feature_pipeline, explainer = load_model()
    print("[OK] Model loaded.\n")

    while True:
        # Get location
        if args.location:
            location = args.location
            args.location = None  # Only use CLI arg once, then interactive
        else:
            location = input("Enter location (PIN code or lat,lon) [q to quit]: ").strip()

        if not location or location.lower() in ("q", "quit", "exit"):
            print("Goodbye!")
            break

        # Fetch data
        print(f"\n[*] Fetching data for '{location}'...")
        try:
            result = fetch_location_data(location)
        except ValueError as e:
            print(f"[ERROR] {e}")
            print("Try a 6-digit Indian PIN code (e.g., 560001) or lat,lon (e.g., 12.97,77.59)\n")
            continue
        except Exception as e:
            print(f"[ERROR] Data fetch failed: {e}\n")
            continue

        # Run prediction
        print("[*] Running model...")
        predictions = predict(
            result.model_input, classifier, yield_model, feature_pipeline, explainer
        )

        # Display
        print_results(
            result.location_info,
            result.soil_data,
            result.weather_data,
            predictions,
            result.data_sources,
        )

        if result.warnings:
            for w in result.warnings:
                print(f"  [!] {w}")
            print()


if __name__ == "__main__":
    main()
