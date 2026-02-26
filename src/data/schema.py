"""
Canonical schema definitions and unit conversion utilities for the
Crop Recommendation System.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import pandas as pd
import numpy as np


# ---------- Canonical column names ----------
CANONICAL_COLUMNS = [
    "sample_id", "date", "latitude", "longitude", "location_id",
    "pH", "EC_dS_m", "OC_pct", "N_kg_ha", "P_kg_ha", "K_kg_ha",
    "S_mg_kg", "zn_ppm", "fe_ppm", "mn_ppm", "cu_ppm",
    "texture", "moisture_pct", "previous_crop", "irrigation", "season",
    "avg_precip_mm", "avg_temp_c", "humidity_pct",
    "target_crop", "yield_kg_ha", "label_source",
]

# Minimal features required for inference (matches Kaggle dataset)
REQUIRED_INFERENCE_FEATURES = [
    "N_kg_ha", "P_kg_ha", "K_kg_ha", "pH",
    "avg_temp_c", "humidity_pct", "avg_precip_mm",
]

# Crop-label mapping (22 crops from Kaggle dataset)
CROP_LABELS: List[str] = sorted([
    "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas",
    "mothbeans", "mungbean", "blackgram", "lentil", "pomegranate",
    "banana", "mango", "grapes", "watermelon", "muskmelon",
    "apple", "orange", "papaya", "coconut", "cotton", "jute", "coffee",
])

CROP_TO_IDX = {c: i for i, c in enumerate(CROP_LABELS)}
IDX_TO_CROP = {i: c for c, i in CROP_TO_IDX.items()}


# ---------- Validation ranges ----------
FEATURE_RANGES = {
    "pH":           (0.0, 14.0),
    "N_kg_ha":      (0.0, 500.0),
    "P_kg_ha":      (0.0, 500.0),
    "K_kg_ha":      (0.0, 500.0),
    "avg_temp_c":   (-10.0, 60.0),
    "humidity_pct":  (0.0, 100.0),
    "avg_precip_mm": (0.0, 5000.0),
    "EC_dS_m":      (0.0, 50.0),
    "OC_pct":       (0.0, 20.0),
    "S_mg_kg":      (0.0, 500.0),
    "zn_ppm":       (0.0, 100.0),
    "fe_ppm":       (0.0, 500.0),
    "mn_ppm":       (0.0, 500.0),
    "cu_ppm":       (0.0, 100.0),
    "moisture_pct":  (0.0, 100.0),
}


# ---------- Column mapping from raw source datasets ----------
KAGGLE_CROP_REC_COLUMN_MAP = {
    "N":           "N_kg_ha",
    "P":           "P_kg_ha",
    "K":           "K_kg_ha",
    "temperature": "avg_temp_c",
    "humidity":    "humidity_pct",
    "ph":          "pH",
    "rainfall":    "avg_precip_mm",
    "label":       "target_crop",
}

INDIA_CROP_PRODUCTION_COLUMN_MAP = {
    "State_Name":    "state",
    "District_Name": "district",
    "Crop_Year":     "year",
    "Season":        "season",
    "Crop":          "target_crop",
    "Area":          "area_ha",
    "Production":    "production_tonnes",
}

KAGGLE_YIELD_COLUMN_MAP = {
    "Crop":            "target_crop",
    "Crop_Year":       "year",
    "Season":          "season",
    "State":           "state",
    "Area":            "area_ha",
    "Production":      "production_tonnes",
    "Annual_Rainfall": "avg_precip_mm",
    "Fertilizer":      "fertilizer_kg_ha",
    "Pesticide":       "pesticide_kg_ha",
    "Yield":           "yield_kg_ha",
}


def normalize_crop_name(name: str) -> str:
    """Normalize crop names to lowercase, stripped, consistent format."""
    if not isinstance(name, str):
        return str(name).lower().strip()
    return name.lower().strip()


def rename_kaggle_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Kaggle Crop Recommendation dataset columns to canonical schema."""
    df = df.rename(columns=KAGGLE_CROP_REC_COLUMN_MAP)
    if "target_crop" in df.columns:
        df["target_crop"] = df["target_crop"].apply(normalize_crop_name)
    return df


def compute_yield(df: pd.DataFrame) -> pd.DataFrame:
    """Compute yield in kg/ha from production (tonnes) and area (ha)."""
    if "production_tonnes" in df.columns and "area_ha" in df.columns:
        df["yield_kg_ha"] = np.where(
            df["area_ha"] > 0,
            (df["production_tonnes"] * 1000.0) / df["area_ha"],
            np.nan,
        )
    return df
