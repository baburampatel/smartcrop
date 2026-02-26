"""
Rule-based baseline crop recommender using agronomic parameter ranges.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


# Expert-defined optimal parameter ranges per crop
# Format: (N_min, N_max, P_min, P_max, K_min, K_max,
#           temp_min, temp_max, hum_min, hum_max,
#           ph_min, ph_max, rain_min, rain_max)
CROP_RULES = {
    "rice":         (60, 110, 30, 65, 30, 55, 18, 30, 70, 95, 5.5, 7.5, 150, 300),
    "maize":        (60, 100, 30, 65, 10, 35, 17, 30, 50, 80, 5.5, 7.5, 50, 120),
    "chickpea":     (20, 60, 50, 85, 65, 100, 12, 25, 10, 25, 6.0, 8.0, 50, 110),
    "kidneybeans":  (10, 35, 50, 85, 10, 35, 14, 27, 15, 35, 5.0, 7.0, 70, 140),
    "pigeonpeas":   (10, 35, 50, 85, 10, 35, 22, 35, 35, 65, 5.0, 7.5, 110, 200),
    "mothbeans":    (10, 35, 35, 65, 10, 35, 22, 35, 35, 65, 5.5, 8.0, 30, 75),
    "mungbean":     (10, 35, 35, 65, 10, 35, 22, 35, 75, 95, 5.5, 7.5, 30, 75),
    "blackgram":    (25, 55, 55, 85, 12, 30, 25, 38, 55, 80, 6.0, 8.0, 45, 90),
    "lentil":       (10, 35, 50, 85, 10, 35, 16, 30, 35, 65, 5.5, 8.0, 30, 70),
    "pomegranate":  (10, 35, 5, 20, 30, 55, 17, 30, 82, 98, 5.5, 7.5, 80, 140),
    "banana":       (80, 130, 60, 95, 40, 65, 23, 33, 73, 90, 5.0, 7.0, 80, 130),
    "mango":        (10, 35, 10, 30, 20, 45, 25, 38, 40, 65, 5.0, 7.0, 65, 130),
    "grapes":       (10, 35, 100, 150, 170, 230, 17, 32, 75, 90, 5.0, 7.0, 50, 90),
    "watermelon":   (85, 120, 10, 25, 40, 60, 20, 33, 78, 93, 5.5, 7.5, 35, 70),
    "muskmelon":    (85, 120, 10, 28, 40, 60, 22, 35, 87, 97, 5.5, 7.5, 15, 40),
    "apple":        (10, 35, 100, 150, 170, 230, 16, 28, 87, 97, 5.0, 7.0, 85, 140),
    "orange":       (10, 35, 5, 18, 5, 18, 16, 28, 87, 97, 6.0, 8.0, 85, 135),
    "papaya":       (35, 70, 45, 80, 40, 65, 28, 40, 87, 97, 5.5, 7.5, 110, 180),
    "coconut":      (10, 35, 5, 18, 20, 45, 23, 33, 90, 100, 5.0, 7.0, 130, 220),
    "cotton":       (100, 145, 32, 60, 12, 30, 18, 30, 73, 90, 6.0, 8.0, 60, 105),
    "jute":         (60, 100, 30, 58, 30, 50, 21, 30, 78, 93, 5.5, 8.0, 140, 210),
    "coffee":       (80, 120, 12, 35, 20, 42, 21, 30, 48, 70, 5.5, 7.5, 125, 195),
}

# Approximate average yields (kg/ha) per crop for rule-based prediction
CROP_AVG_YIELDS = {
    "rice": 2500, "maize": 2800, "chickpea": 1000, "kidneybeans": 1200,
    "pigeonpeas": 800, "mothbeans": 400, "mungbean": 600, "blackgram": 600,
    "lentil": 800, "pomegranate": 8000, "banana": 30000, "mango": 7000,
    "grapes": 20000, "watermelon": 25000, "muskmelon": 15000, "apple": 12000,
    "orange": 10000, "papaya": 40000, "coconut": 8000, "cotton": 500,
    "jute": 2200, "coffee": 800,
}


def compute_suitability_score(
    sample: dict, crop: str, rules: dict = CROP_RULES
) -> float:
    """
    Compute a suitability score (0-1) for a given crop and soil sample.
    Uses a fuzzy membership approach — score = fraction of features
    within the crop's optimal range.
    """
    if crop not in rules:
        return 0.0

    bounds = rules[crop]
    n_lo, n_hi, p_lo, p_hi, k_lo, k_hi = bounds[:6]
    t_lo, t_hi, h_lo, h_hi, ph_lo, ph_hi, r_lo, r_hi = bounds[6:]

    checks = [
        (sample.get("N_kg_ha", 0), n_lo, n_hi),
        (sample.get("P_kg_ha", 0), p_lo, p_hi),
        (sample.get("K_kg_ha", 0), k_lo, k_hi),
        (sample.get("avg_temp_c", 25), t_lo, t_hi),
        (sample.get("humidity_pct", 50), h_lo, h_hi),
        (sample.get("pH", 7), ph_lo, ph_hi),
        (sample.get("avg_precip_mm", 100), r_lo, r_hi),
    ]

    score = 0.0
    for val, lo, hi in checks:
        if lo <= val <= hi:
            # Within range: full score
            score += 1.0
        else:
            # Partial credit based on distance from range
            if val < lo:
                dist = (lo - val) / max(lo, 1)
            else:
                dist = (val - hi) / max(hi, 1)
            score += max(0, 1.0 - dist)

    return score / len(checks)


def predict_rule_based(sample: dict, top_k: int = 3) -> List[Dict]:
    """
    Rule-based crop recommendation.
    Returns top-k crops with suitability scores and expected yields.
    """
    scores = {}
    for crop in CROP_RULES:
        scores[crop] = compute_suitability_score(sample, crop)

    sorted_crops = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    total = sum(s for _, s in sorted_crops[:top_k]) or 1.0

    results = []
    for crop, score in sorted_crops[:top_k]:
        results.append({
            "crop": crop,
            "probability": round(score / total, 4),
            "expected_yield_kg_ha": CROP_AVG_YIELDS.get(crop, 0),
            "explanation": [
                {"feature": "rule_based", "contribution": round(score, 4)}
            ],
        })
    return results


def evaluate_baseline(df: pd.DataFrame) -> Dict:
    """Evaluate baseline rule-based recommender on a dataset."""
    correct_top1 = 0
    correct_top3 = 0
    total = len(df)

    for _, row in df.iterrows():
        sample = row.to_dict()
        preds = predict_rule_based(sample, top_k=3)
        pred_crops = [p["crop"] for p in preds]
        true_crop = sample.get("target_crop", "")

        if pred_crops and pred_crops[0] == true_crop:
            correct_top1 += 1
        if true_crop in pred_crops:
            correct_top3 += 1

    return {
        "top1_accuracy": round(correct_top1 / max(total, 1), 4),
        "top3_accuracy": round(correct_top3 / max(total, 1), 4),
        "total_samples": total,
    }
