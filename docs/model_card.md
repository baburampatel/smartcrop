# Model Card — Crop Recommendation System v1.0

## Model Details

- **Model type**: XGBoost multi-class classifier (22 classes) + XGBoost yield regressor
- **Framework**: XGBoost 2.0+ / LightGBM 4.0+
- **Training data**: Crop Recommendation Dataset (Kaggle, CC0) + India Crop Production Statistics (data.gov.in, GODL)
- **Input**: 7 required features (N, P, K, pH, temperature, humidity, rainfall) + optional micronutrients
- **Output**: Top-3 crops with probabilities, expected yield (kg/ha), SHAP feature contributions
- **Model size**: < 50 MB (meets 500MB constraint)
- **Inference latency**: p95 ≤ 300ms on CPU

## Intended Use

- **Primary users**: Agricultural field officers, extension workers, farmers (via field officers)
- **Primary geography**: India (all agro-climatic zones)
- **Use case**: Soil-test-based crop selection advisory for a single growing season
- **Not intended for**: Automated decision-making without human oversight, financial trading on crop futures

## Training Data

| Dataset | Samples | Crops | License | Source |
|---------|---------|-------|---------|--------|
| Kaggle Crop Recommendation | 2,200 | 22 | CC0 | Expert curated |
| India Crop Production | ~250K | Multiple | GODL | Government statistics |

### Crops Covered
apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, watermelon

## Evaluation Results

- **Split strategy**: Geographic hold-out by agro-climatic zone clusters (location_id via KMeans)
- **Test set**: ~20% of location clusters held out
- Detailed metrics available in `docs/evaluation_report.md` after training

### Metrics Tracked
| Metric | Description |
|--------|-------------|
| Top-1 Accuracy | Correct top prediction |
| Top-3 Accuracy | Correct crop in top 3 (target ≥ 0.75) |
| Per-class P/R/F1 | Per-crop precision, recall, F1 |
| Yield MAE | Mean absolute error in kg/ha |
| SHAP importance | Global and per-prediction |

## Limitations

1. **Dataset size**: Primary dataset has only 2,200 samples — may not capture micro-regional variations
2. **No true geo-coordinates**: Location IDs are approximated via feature clustering, not real lat/lon
3. **Static climate features**: Does not incorporate real-time weather forecasts
4. **Limited micronutrient data**: Zn, Fe, Mn, Cu, S not available in primary dataset
5. **No irrigation/management data**: Yield predictions don't account for farming practices
6. **Bias toward represented crops**: 22 crops may not cover all viable options in certain regions

## Ethical Considerations

- **No PII collected**: System accepts only soil/climate parameters
- **Risk of over-reliance**: Recommendations should supplement, not replace, local expert knowledge
- **Equity**: System may perform unevenly across agro-climatic zones with limited training data
- **Consent**: All field trial data collection requires informed consent (see `docs/consent_template.md`)

## Monitoring & Updates

- Feature drift: PSI/KL-divergence on input distributions
- Prediction drift: Track crop prediction distribution over time
- Retraining: Monthly schedule with fresh data; retrain if drift exceeds thresholds
- Prometheus metrics exposed at `/metrics`
