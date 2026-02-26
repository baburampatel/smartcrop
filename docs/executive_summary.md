# Executive Summary — Crop Recommendation System

## Problem

Indian farmers face a critical challenge: selecting the optimal crop for their specific soil and climate conditions. Suboptimal crop selection leads to reduced yields, wasted inputs, and lower incomes. While soil testing is increasingly available through the Soil Health Card scheme, translating test results into actionable crop recommendations remains a gap.

## Solution

We built a **production-grade ML-powered crop recommendation system** that:

1. Takes soil lab values (N, P, K, pH) and climate data (temperature, humidity, rainfall) as input
2. Returns **top-3 crop recommendations** ranked by suitability probability
3. Provides **expected yield estimates** (kg/ha) for each recommended crop
4. Includes **SHAP-based explanations** showing which factors drove each recommendation

## Key Results

| Metric | Target | Details |
|--------|--------|---------|
| Crops covered | 22 | Major Indian crops (cereals, pulses, oilseeds, fruits, cash crops) |
| Model | XGBoost | Multi-class classifier with Optuna-tuned hyperparameters |
| Top-3 accuracy | ≥ 0.75 target | Evaluated on geographically held-out test set |
| Inference latency | p95 ≤ 300ms | Measured on CPU |
| Model artifact | < 50 MB | Well within 500 MB limit |

## Technical Architecture

```
Soil Sample → ETL Pipeline → Feature Engineering → XGBoost Classifier → Top-3 Crops
                                                 → Yield Regressor   → Expected Yield
                                                 → SHAP Explainer    → Feature Contributions
```

- **API**: FastAPI with `/predict` endpoint
- **Deployment**: Docker + Terraform (AWS EC2, ap-south-1)
- **Monitoring**: Prometheus metrics, CloudWatch logs
- **MLOps**: MLflow tracking, GitHub Actions CI/CD, monthly retraining

## Datasets

Three publicly licensed datasets totaling ~260K records:
- Kaggle Crop Recommendation (2,200 expert-curated samples, CC0)
- India Crop Production Statistics (data.gov.in, GODL)
- Kaggle Agricultural Crop Yield (1997-2020, CC0)

No restricted data used. No PII collected. All licenses documented in `data/data_catalog.json`.

## Deliverables

| Artifact | Status |
|----------|--------|
| Data catalog & ETL pipeline | ✅ Complete |
| Data validation suite (8 checks) | ✅ Complete |
| Labeling strategy & bias analysis | ✅ Documented |
| Feature engineering pipeline | ✅ Complete |
| Baseline rule-based recommender | ✅ Complete |
| XGBoost classifier + yield model | ✅ Complete |
| SHAP explainability | ✅ Complete |
| FastAPI inference service | ✅ Complete |
| Field officer web UI | ✅ Complete |
| Docker + docker-compose | ✅ Complete |
| Terraform (AWS) | ✅ Complete |
| CI/CD (GitHub Actions) | ✅ Complete |
| Monitoring (Prometheus) | ✅ Complete |
| Tests (unit + API + latency) | ✅ Complete |
| Model card | ✅ Complete |
| Field trial plan (50-200 farmers) | ✅ Complete |
| Consent templates & PII redaction | ✅ Complete |

## Next Steps

1. **Field validation**: Execute the A/B pilot plan with 100-200 farmers across 4-6 districts
2. **Data enrichment**: Integrate micronutrient data (Zn, Fe, Mn, Cu) and irrigation information
3. **Real-time weather**: Connect to IMD/weather APIs for dynamic climate features
4. **Scale evaluation**: Expand to additional crops and agro-climatic zones
5. **Mobile application**: Build offline-capable mobile app for field officers
