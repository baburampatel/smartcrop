# Crop Recommendation System

A production-grade ML system that takes soil sample data (lab values + contextual data) and returns **top-3 crop recommendations** with **expected yield** and **SHAP-based explanations**. Focused on Indian agriculture.

## ✨ Features

- **Multi-crop classification** — XGBoost classifier trained on 22 India crops
- **Top-3 recommendations** with calibrated probabilities
- **Per-prediction SHAP explanations** for interpretability
- **Yield prediction** per recommended crop
- **Rule-based baseline** for comparison and fallback
- **FastAPI inference service** with p95 latency ≤ 300ms
- **Docker containerization** with health checks
- **Terraform infrastructure** (AWS EC2 + CloudWatch)
- **MLflow experiment tracking** and model registry
- **Data validation suite** (8 automated checks)
- **Web UI** for field officers
- **CI/CD pipeline** (GitHub Actions)
- **Prometheus metrics** for drift/latency monitoring

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run ETL pipeline
```bash
python scripts/ingest.py --data-dir data
```

### 3. Train models
```bash
python scripts/train.py --data-dir data --model-type xgboost
# With hyperparameter tuning:
python scripts/train.py --data-dir data --tune --tune-trials 50
```

### 4. Evaluate
```bash
python scripts/evaluate.py --data-dir data --models-dir models
```

### 5. Run API server
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### 6. Make a prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"N":90,"P":42,"K":43,"temperature":20.87,"humidity":82.0,"ph":6.5,"rainfall":202.9}'
```

## 🐳 Docker

```bash
# Build and run
docker-compose up -d

# API at http://localhost:8000
# MLflow at http://localhost:5000
```

## ☁️ Cloud Deployment

```bash
cd terraform
terraform init
terraform plan -var="key_pair_name=your-key"
terraform apply
```

## 📂 Project Structure

```
├── src/
│   ├── data/         — Schema, validation, feature engineering
│   ├── models/       — Baseline, XGBoost classifier, yield model
│   ├── explain/      — SHAP explainability
│   └── api/          — FastAPI application
├── scripts/          — ETL, training, evaluation entry points
├── tests/            — Unit, API, and latency tests
├── ui/               — Field officer web interface
├── models/           — Saved model artifacts
├── data/             — Raw + processed data + catalog
├── terraform/        — AWS infrastructure
├── docs/             — Model card, field trial, consent forms
└── .github/workflows — CI pipeline
```

## 📊 API Response Schema

```json
{
  "top_3": [
    {
      "crop": "rice",
      "probability": 0.85,
      "expected_yield_kg_ha": 2500.0,
      "explanation": [
        {"feature": "avg_precip_mm", "contribution": 0.23},
        {"feature": "N_kg_ha", "contribution": 0.18}
      ]
    }
  ],
  "model_version": "1.0.0",
  "data_checksum": "abc123..."
}
```

## 🧪 Testing

```bash
# All tests
pytest tests/ -v

# Latency benchmark
pytest tests/test_latency.py -v -s

# Data validation
python -m src.data.validation
```

## 📜 Datasets

All datasets are publicly licensed (CC0 / GODL). See `data/data_catalog.json` for full details.

| Dataset | Source | License | Records |
|---------|--------|---------|---------|
| Crop Recommendation | Kaggle | CC0 | 2,200 |
| India Crop Production | data.gov.in | GODL | ~250K |
| Crop Yield States | Kaggle | CC0 | ~10K |

## 📄 Documentation

- [Model Card](docs/model_card.md)
- [Data Catalog](docs/data_catalog.md)
- [Field Trial Plan](docs/field_trial_plan.md)
- [Executive Summary](docs/executive_summary.md)
- [Consent Template](docs/consent_template.md)
- [Evaluation Report](docs/evaluation_report.md)
- [Labeling Strategy](notebooks/labeling.md)

## License

This project is for research and development purposes. See individual dataset licenses in `data/data_catalog.json`.
