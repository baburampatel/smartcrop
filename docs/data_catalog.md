# Data Catalog

Complete inventory of all datasets used in the Crop Recommendation System. All datasets are legally obtainable under open/public licenses.

## Datasets

### 1. Crop Recommendation Dataset (Primary)

| Field | Value |
|-------|-------|
| **Source** | Kaggle — Atharva Ingle |
| **URL** | https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset |
| **License** | CC0: Public Domain |
| **Retrieval Date** | 2026-02-24 |
| **Format** | CSV |
| **Records** | 2,200 (100 per crop × 22 crops) |
| **Contact** | Kaggle community |

**Variables:**

| Variable | Unit | Type | Description |
|----------|------|------|-------------|
| N | kg/ha | float | Nitrogen content ratio |
| P | kg/ha | float | Phosphorous content ratio |
| K | kg/ha | float | Potassium content ratio |
| temperature | °C | float | Temperature |
| humidity | % | float | Relative humidity |
| ph | — | float | Soil pH value |
| rainfall | mm | float | Rainfall |
| label | — | string | Recommended crop (22 classes) |

---

### 2. India Crop Production Statistics

| Field | Value |
|-------|-------|
| **Source** | Open Government Data Platform India |
| **URL** | https://data.gov.in/resource/district-wise-season-wise-crop-production-statistics |
| **License** | Government Open Data License — India (GODL) |
| **Retrieval Date** | 2026-02-24 |
| **Format** | CSV |
| **Records** | ~250,000+ |
| **Contact** | Ministry of Agriculture, Government of India |

**Variables:**

| Variable | Unit | Type | Description |
|----------|------|------|-------------|
| State_Name | — | string | Indian state |
| District_Name | — | string | District |
| Crop_Year | year | int | Harvest year |
| Season | — | string | Kharif / Rabi / Whole Year |
| Crop | — | string | Crop name |
| Area | hectares | float | Cultivated area |
| Production | tonnes | float | Total production |

---

### 3. Agricultural Crop Yield in Indian States

| Field | Value |
|-------|-------|
| **Source** | Kaggle |
| **URL** | https://www.kaggle.com/datasets/akshatgupta7/crop-yield-in-indian-states-dataset |
| **License** | CC0: Public Domain |
| **Retrieval Date** | 2026-02-24 |
| **Format** | CSV |
| **Records** | ~10,000+ |
| **Contact** | Kaggle community |

**Variables:**

| Variable | Unit | Type | Description |
|----------|------|------|-------------|
| Crop | — | string | Crop type |
| Crop_Year | year | int | Year |
| Season | — | string | Cropping season |
| State | — | string | Indian state |
| Area | hectares | float | Cultivated area |
| Production | tonnes | float | Total production |
| Annual_Rainfall | mm | float | Annual rainfall |
| Fertilizer | kg/ha | float | Fertilizer usage |
| Pesticide | kg/ha | float | Pesticide usage |
| Yield | kg/ha | float | Computed yield |

## Redistribution Notes

- **CC0 datasets**: No restrictions on redistribution
- **GODL (India)**: Free to use, share, and adapt with attribution to Government of India
- No PII present in any dataset
- All datasets are aggregate/statistical — no individual farmer data
