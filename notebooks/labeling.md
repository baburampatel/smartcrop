# Labeling Strategy

## Overview

This document describes how crop labels (target_crop) and yield labels (yield_kg_ha) are assigned to each soil sample in the training dataset.

## Label Source: Expert Crop Recommendations

The primary dataset (Kaggle Crop Recommendation Dataset) contains crop labels curated by agricultural experts. Each sample represents a set of soil and climate conditions with the **expert-recommended optimal crop**.

### Labeling Approach
- **Source**: Published agronomic guidelines and expert knowledge
- **Method**: For each combination of soil parameters (N, P, K, pH) and climate conditions (temperature, humidity, rainfall), experts identified the most suitable crop from a set of 22 common Indian crops
- **Label field**: `label_source = "kaggle_expert_recommendation"`

## Yield Labels

Yield data is sourced from the India Crop Production Statistics (data.gov.in) and integrated via the ETL pipeline:

1. **District-level yield** = Production (tonnes) × 1000 / Area (hectares) = kg/ha
2. **Per-crop median yield** is computed across all districts and years (2015-2022)
3. Each training sample receives the **median yield** for its assigned crop

### Limitations
- Yield is at district aggregate level, not field-level
- No connection between specific soil samples and specific yield observations
- Yield variation due to management, irrigation, variety is not captured

## Bias Analysis

### Known Biases

| Bias Type | Description | Mitigation |
|-----------|-------------|------------|
| **Selection bias** | Dataset covers 22 crops; many crops are excluded | Document coverage, support fallback for unknown crops |
| **Class balance** | Exactly 100 samples per crop (balanced by design) | No SMOTE needed for this dataset |
| **Geographic bias** | No explicit geographic tagging; features imply agro-climatic zone | Cluster-based location_id approximation |
| **Temporal bias** | No temporal variation; single snapshot | Could be improved with multi-season data |
| **Expert bias** | Labels reflect expert opinion, not field validation | Field trial plan addresses this (see docs/field_trial_plan.md) |

### Crop Distribution

All 22 crops have exactly 100 samples each (2,200 total) — perfectly balanced.

### Feature Distribution Analysis

- **N_kg_ha**: Range [0, ~155], bimodal distribution (low-N pulses vs high-N cereals/cotton)
- **P_kg_ha**: Range [0, ~150], wide distribution
- **K_kg_ha**: Range [0, ~230], heavy right tail (grapes, apple)
- **pH**: Range [3.5, 9.5], centered ~6.5
- **Temperature**: Range [10, 45°C], approximately normal
- **Humidity**: Range [10, 100%], bimodal
- **Rainfall**: Range [15, 300mm], right-skewed

## Expert-Corrected Labeling Procedure

If farmer-choice labels were used instead of expert recommendations, the following procedure would be applied:

1. **Collect farmer labels**: Record what farmers actually planted
2. **Compare with expert recommendation**: Identify discrepancies
3. **Expert review panel**: 3 agronomists independently review discrepant cases
4. **Majority vote**: Accept label agreed upon by 2+ experts
5. **Document disagreements**: Track cases where experts disagree for further study
6. **Bias quantification**: Report agreement rate between farmer choices and expert labels

This procedure is not currently needed since the primary dataset uses expert-curated labels.
