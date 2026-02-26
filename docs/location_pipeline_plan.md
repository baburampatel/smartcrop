# Location-Agnostic Crop Recommendation Pipeline Plan

> **LOCATION** = `{LOCATION}` (runtime parameter — Indian PIN code or `lat,lon`)
> **TARGET_CROPS** = `rice,maize,wheat,soybean,groundnut,sugarcane`
> **RADIUS_KM** = `50` | **MAX_BUDGET** = `$20/sample` | **PRIORITIZE** = `data_quality`
> **FIELD_TRIAL_SIZE** = `50`

---

## SUMMARY (Human-Friendly)

1. **Best pipeline**: **Primary (Lab + SoilGrids + Sentinel-2)** — combines ground-truth lab soil tests with ISRIC SoilGrids 250m modeled data and Sentinel-2 NDVI proxies for full feature coverage.
2. **Expected confidence**: **High** — Top-3 accuracy ≥ 0.85 with ≥ 500 geo-referenced samples; existing model already achieves 95.21% on Kaggle benchmark.
3. **Estimated cost**: **$800–$2,500** for MVP (data acquisition + training + FastAPI + Docker), depending on lab test count.
4. **Timeline**: **4–8 weeks** from soil sample collection to deployed endpoint.
5. **Fallback**: Secondary pipeline uses Soil Health Card + remote sensing only (zero lab cost, moderate confidence).

---

## A. DATA SOURCES — LOCATE & ENUMERATE

### A.1 Public Soil Test / Soil Health Card Data

| Source | URL | License | Variables | Last Update | Density in RADIUS_KM |
|--------|-----|---------|-----------|-------------|----------------------|
| **Soil Health Card Portal** | `https://soilhealth.dac.gov.in` | Government (public view, no bulk API) | pH, EC, OC, N, P, K, S, Zn, Fe, Cu, Mn, B, texture | Ongoing (cards issued since 2015) | High in most districts (~1 card per 2.5 ha target) |
| **Google SHC Scraper Dataset** | `https://github.com/google-research-datasets/india-soil-health-card` | Apache 2.0 (scraper); data is GOI | Same as above | 2020 (scraper last run) | Varies by state; bulk metadata available |
| **Kaggle Crop Recommendation** | `https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset` | CC0 | N, P, K, temperature, humidity, pH, rainfall, crop label | 2021 | 2,200 samples (no location tagging) |
| **India Crop Production (data.gov.in)** | `https://data.gov.in/resource/district-wise-season-wise-crop-production-statistics` | GODL | State, District, Crop, Year, Season, Area (ha), Production (t) | 2020 | ~250K records, district-level |

### A.2 Global / Regional Soil & Remote-Sensing Sources

| Source | Endpoint / Method | Resolution | Variables | Programmatic Access |
|--------|-------------------|------------|-----------|---------------------|
| **ISRIC SoilGrids v2.0** | `https://rest.isric.org/soilgrids/v2.0/properties/query?lat={lat}&lon={lon}` | 250m | clay, sand, silt, soc, phh2o, cec, bdod, cfvo, nitrogen, ocd | REST API (currently paused — fallback: WCS/GDAL raster download) |
| **FAO Global Soil Map** | `https://data.apps.fao.org/map/catalog/srv/eng/catalog.search` | 1km | Soil type, texture class, pH | OGC WMS/WFS |
| **NASA SMAP L4** | `https://n5eil01u.ecs.nsidc.org/SMAP/` | 9km | Surface/root-zone soil moisture | Earthdata Login + OPeNDAP |
| **Sentinel-2 (via GEE)** | Google Earth Engine `ee.ImageCollection('COPERNICUS/S2_SR')` | 10m | NDVI, EVI, moisture proxies (NDMI, NDWI) | GEE Python API (`earthengine-api`) |
| **MODIS Land Surface Temp** | GEE `ee.ImageCollection('MODIS/061/MOD11A2')` | 1km | Day/Night LST | GEE Python API |
| **OpenLandMap** | `https://openlandmap.org` WCS | 250m | Soil texture, pH, SOC, bulk density | WCS endpoint (public) |

### A.3 Local Soil Testing Labs & Extension Offices

> **Resolved at runtime** for each `{LOCATION}`. Methodology:

1. Query [Soil Health Card Portal → Lab Directory](https://soilhealth.dac.gov.in/LabDirectory) by state/district.
2. Search `"soil testing laboratory" near {LOCATION}` on Google Maps API.
3. Cross-reference with [ICAR KVK directory](https://kvk.icar.gov.in/) for Krishi Vigyan Kendras within `RADIUS_KM`.

**Typical lab profile (template)**:
- **Name**: State Soil Testing Laboratory / KVK Lab / SAU Lab
- **Cost**: ₹200–₹500/sample (~$2.50–$6.25 USD) for N-P-K-pH-EC-OC; ₹800–₹1500 (~$10–$19) for micronutrients
- **Turnaround**: 7–21 days (government); 3–7 days (private)
- **Online results**: Some state labs publish on SHC portal

### A.4 Market & Weather Data

| Source | URL / API | Fields | Frequency |
|--------|-----------|--------|-----------|
| **Agmarknet** | `https://agmarknet.gov.in` (API via `data.gov.in` Catalog API) | Commodity, Market, State, Arrival (T), Min/Max/Modal Price (₹/q) | Daily |
| **data.gov.in Agmarknet** | `https://api.data.gov.in/resource/{resource_id}?api-key={key}&format=json&filters[state]={state}` | Same | Daily (API key required — free registration) |
| **IMD (India Met Department)** | `https://mausam.imd.gov.in` (no public API; data via `data.gov.in` datasets) | Rainfall, Temperature, Humidity | Daily/Monthly |
| **Open-Meteo** | `https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=...&daily=temperature_2m_max,...` | Temp, Humidity, Precipitation, Wind, Solar | Hourly/Daily (free, no key) |
| **Meteostat** | `https://meteostat.p.rapidapi.com/point/daily?lat={lat}&lon={lon}` | Same | Daily (free tier: 500 req/month) |

### A.5 Crowdsourced / Farmer Data Portals

| Platform | Access | Data |
|----------|--------|------|
| **Soil Health Card Mobile App** | Public per-card lookup (farmer Aadhaar/mobile required) | Individual SHC results |
| **Kisan Suvidha** | Public app | Weather, market prices, dealer info (no soil data) |
| **mKisan / Kisan Call Centre** | Public | Advisory messages (text/voice) |
| **Digital Green / CGIAR** | Registration + MoU for research | Farmer practice videos, crop adoption data |
| **Local NGOs / FPOs** | Consent-based partnership | Field-level yield, soil test, practice data |

---

## B. MECHANICS ASSESSMENT

### B.1 Soil Health Card Portal (Scraper)

| Dimension | Assessment |
|-----------|------------|
| **Data quality** | **High** — Lab-tested values (N, P, K, pH, EC, OC, micronutrients); standardized protocol across 488+ labs |
| **Permission** | **Registration required** — Individual card lookup needs farmer mobile/Aadhaar; bulk scraping uses Google's Apache-licensed scraper |
| **Cost & time** | Free (portal is public); scraping 10K cards takes ~2–4 hours |
| **Programmatic access** | Google scraper: `python scraper.py --state=KARNATAKA --district=BANGALORE`; or manual portal search |
| **Suitability** | **Excellent** — Direct lab data; best proxy for ground-truth soil features |

### B.2 ISRIC SoilGrids v2.0

| Dimension | Assessment |
|-----------|------------|
| **Data quality** | **Medium** — Modeled at 250m from global training set; may not capture local Indian soil variability well |
| **Permission** | **Public** — CC-BY 4.0; no registration for REST API |
| **Cost & time** | Free; API response in <1s per query |
| **Programmatic access** | `curl "https://rest.isric.org/soilgrids/v2.0/properties/query?lat=12.97&lon=77.59&property=phh2o&depth=0-5cm&value=mean"` |
| **Suitability** | **Good fallback** — Use when lab data unavailable; provides clay/sand/silt/SOC/pH at any point |

### B.3 Sentinel-2 via Google Earth Engine

| Dimension | Assessment |
|-----------|------------|
| **Data quality** | **High** — 10m multispectral; 5-day revisit; NDVI/EVI are proven crop health indicators |
| **Permission** | **Registration required** — Free GEE account (research/non-commercial); commercial use needs GEE commercial license |
| **Cost & time** | Free for research; ~5s per imagery extract |
| **Programmatic access** | `ee.ImageCollection('COPERNICUS/S2_SR').filterBounds(ee.Geometry.Point(lon,lat)).filterDate(start,end).map(compute_ndvi)` |
| **Suitability** | **Complementary** — NDVI/phenology features improve yield prediction; not a direct soil substitute |

### B.4 Kaggle Crop Recommendation Dataset (Existing)

| Dimension | Assessment |
|-----------|------------|
| **Data quality** | **Medium** — Expert-curated but lacks geo-tagging; 22 crops × 100 samples each |
| **Permission** | **Public** — CC0 |
| **Cost & time** | Free; instant download |
| **Programmatic access** | `kaggle datasets download atharvaingle/crop-recommendation-dataset` |
| **Suitability** | **Good baseline** — Already powering the existing model (Top-3 = 95.2%); insufficient for location-specific fine-tuning alone |

### B.5 Agmarknet Market Prices

| Dimension | Assessment |
|-----------|------------|
| **Data quality** | **High** — Official mandi prices from 3,000+ regulated markets |
| **Permission** | **Registration required** — Free API key from data.gov.in |
| **Cost & time** | Free; API quota varies |
| **Programmatic access** | `curl "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={KEY}&format=json&filters[State.keyword]=Karnataka&filters[Commodity]=Rice"` |
| **Suitability** | **Supplementary** — Market price signals help rank crop profitability; not a soil input |

### B.6 Open-Meteo Weather Archive

| Dimension | Assessment |
|-----------|------------|
| **Data quality** | **High** — Reanalysis-backed (ERA5); validated against station data |
| **Permission** | **Public** — No API key needed; CC-BY 4.0 |
| **Cost & time** | Free; <1s per query |
| **Programmatic access** | `curl "https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2024-01-01&end_date=2024-12-31&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"` |
| **Suitability** | **Essential** — Temperature, humidity, rainfall are core model features |

---

## C. DATA COLLECTION & FALLBACK STRATEGY (Ranked Pipelines)

### Pipeline 1: PRIMARY — Lab + SoilGrids + Sentinel-2 + Weather

**Required inputs (exact features and units)**:
| Feature | Unit | Source |
|---------|------|--------|
| pH | 1–14 | Lab test |
| EC | dS/m | Lab test |
| N_kg_ha | kg/ha | Lab test |
| P_kg_ha | kg/ha | Lab test |
| K_kg_ha | kg/ha | Lab test |
| OC_pct | % | Lab test |
| texture_class | categorical | Lab test or SoilGrids (clay/sand/silt %) |
| avg_temp_c | °C | Open-Meteo |
| humidity_pct | % | Open-Meteo |
| avg_precip_mm | mm | Open-Meteo |
| ndvi_mean | 0–1 | Sentinel-2 via GEE |
| latitude | decimal degrees | GPS/geocoded |
| longitude | decimal degrees | GPS/geocoded |
| season | Kharif/Rabi/Zaid | User input or derived from month |

**Data acquisition mechanics**:
1. Geocode `{LOCATION}` → lat/lon (PIN → district centroid via India Post API or lookup table).
2. Collect 10–50 soil samples within `RADIUS_KM` → send to nearest lab (A.3).
3. Fetch SoilGrids modeled values for each sample point (gap-fill missing lab variables).
4. Fetch 3-year weather history from Open-Meteo for the location.
5. Extract Sentinel-2 NDVI time series for sample locations via GEE.
6. Pull district-level yield history from data.gov.in for TARGET_CROPS.
7. Pull Agmarknet mandi prices for nearest market for economic ranking.

**ETL outline**:
1. **Harmonize units**: Soil lab reports vary — normalize N/P/K to kg/ha; convert EC to dS/m; standardize pH scale.
2. **Unit conversion**: SoilGrids phh2o is in pH×10 → divide by 10; clay/sand/silt in g/kg → /10 for %.
3. **Missing data imputation**: For missing lab values (e.g., micronutrients), impute from SoilGrids or district-median from SHC data. For missing weather, use climatological mean from Open-Meteo.
4. **Feature engineering**: Compute N:P ratio, N:K ratio, pH bin (acidic/neutral/alkaline), NDVI seasonal amplitude, growing-degree-days.
5. **Merge**: Join soil + weather + NDVI + yield + price on (location_id, season, crop).

**Per-sample cost & time**: ₹300–₹500 lab ($4–$6) + $0 API = **~$5/sample**; turnaround 7–14 days (lab bottleneck).

**Confidence**: **HIGH** — Lab ground-truth + remote sensing + weather covers all key agronomic drivers. With 500+ samples, Top-3 accuracy ≥ 0.85 expected.

---

### Pipeline 2: SECONDARY — Soil Health Card (Scraped) + Remote Sensing + Weather

**Required inputs**: Same features as Primary, but **no new lab sampling** — all soil data from existing SHC records.

**Data acquisition mechanics**:
1. Geocode `{LOCATION}` → state + district.
2. Scrape SHC portal for all cards in target district(s) using Google Research scraper.
3. Parse card images (OCR) or metadata JSON for pH, N, P, K, OC, EC.
4. Fetch weather & NDVI as in Primary pipeline.
5. Match SHC records to district yield data.

**ETL outline**: Same harmonization as Primary; additional OCR-error cleaning step for scraped values (range checks, outlier removal).

**Per-sample cost & time**: **$0/sample** (existing data); scraping & processing 2–5 days.

**Confidence**: **MEDIUM-HIGH** — SHC data is lab-tested but: (a) 2–4 year old, (b) location precision is village-level not GPS, (c) OCR may introduce errors. Top-3 accuracy ≥ 0.78 expected with 1,000+ scraped records.

---

### Pipeline 3: TERTIARY — SoilGrids + Kaggle Baseline + Weather (Zero-Cost)

**Required inputs**: Same features, but **all soil values from SoilGrids modeled data** + Kaggle training set for classification.

**Data acquisition mechanics**:
1. Geocode `{LOCATION}` → lat/lon.
2. Query SoilGrids REST API (or WCS raster) for soil properties at location.
3. Map SoilGrids clay/sand/silt/pH/SOC to N/P/K approximations using pedotransfer functions.
4. Fetch weather from Open-Meteo.
5. Use pre-trained XGBoost model (existing) with SoilGrids values as input.
6. Yield estimates from district-level historical data only.

**ETL outline**: Pedotransfer function for N (f(SOC, texture)), P/K from regional soil survey correlations.

**Per-sample cost & time**: **$0/sample**; <1 minute per location (all API-based).

**Confidence**: **MEDIUM** — SoilGrids resolution (250m) misses field-level variability; N/P/K are approximated not measured. Top-3 accuracy ~0.70–0.75.

---

## D. LOCAL SAMPLING & SENSOR MECHANICS

### D.1 Soil Sampling Protocol (Tailored for Indian Agriculture)

1. **Define sampling unit**: One composite sample per 1–2 hectare plot.
2. **Select subsampling points**: Walk a zigzag (or "W") pattern across the field; collect from **15–20 subsampling points** per composite.
3. **Depth**: Use a soil auger or khurpi to collect from **0–15 cm** (plough layer) and optionally **15–30 cm** (subsoil).
4. **Avoid edges**: Stay ≥10m from field boundaries, bunds, tree canopy, manure heaps.
5. **Mix**: Combine all subsamples in a clean plastic bucket; mix thoroughly by hand.
6. **Quarter**: Use quartering method — flatten into a circle, divide into 4, discard 2 opposite quarters, repeat until ~500g remains.
7. **Air dry**: Spread on clean paper in shade for 24–48 hours.
8. **Pack**: Place in a clean cloth or poly bag; label clearly.
9. **Send**: Deliver to nearest lab within 7 days.

### D.2 Required Field Kit (Estimated Prices)

| Item | Approx. Price (INR) | Approx. Price (USD) |
|------|---------------------|---------------------|
| Soil auger (manual, 30cm) | ₹800–₹1,500 | $10–$19 |
| Khurpi (hand trowel) | ₹100–₹200 | $1.25–$2.50 |
| Plastic bucket (10L) | ₹150–₹250 | $2–$3 |
| Sample bags (poly, 50 pcs) | ₹200–₹300 | $2.50–$3.75 |
| Permanent markers (5 pcs) | ₹100 | $1.25 |
| GPS device (or phone w/ GPS app) | ₹0 (use phone) | $0 |
| Field data form (printed, 50 copies) | ₹200 | $2.50 |
| Measuring tape (30m) | ₹300 | $3.75 |
| **Total kit cost** | **₹1,850–₹2,850** | **$23–$36** |

### D.3 Recommended Lab Form Metadata

| Field | Example |
|-------|---------|
| `sample_id` | `{LOCATION}-{YYYYMMDD}-{NNN}` |
| `gps_lat` | 12.9716 |
| `gps_lon` | 77.5946 |
| `date_collected` | 2026-02-26 |
| `depth_cm` | 0–15 |
| `farmer_id` | Optional (anonymized code if consented) |
| `previous_crop` | Rice / Maize / Fallow |
| `irrigation_type` | Rainfed / Borewell / Canal |
| `season` | Kharif / Rabi / Zaid |
| `field_area_ha` | 1.5 |

### D.4 Low-Cost Sensor Alternatives

| Device | Measures | Accuracy vs Lab | Approx. Price | Buy Link Example |
|--------|----------|-----------------|---------------|------------------|
| Portable pH meter (Atree PH20) | pH | ±0.1 pH (good) | ₹1,500–₹3,000 ($19–$38) | Amazon.in search "soil pH meter" |
| EC/TDS meter | EC (dS/m) | ±5% (good) | ₹800–₹1,500 ($10–$19) | Amazon.in |
| Rapid N-P-K test kit (LaMotte) | N, P, K colorimetric | ±20% (low-moderate) | ₹3,000–₹5,000 ($38–$63) | IndiaMART |
| Soil moisture sensor (capacitive) | Volumetric moisture | ±3% (moderate) | ₹300–₹600 ($4–$8) | Amazon.in |
| Handheld NDVI sensor (GreenSeeker) | NDVI | ±0.02 (high) | ₹80,000+ ($1,000+) | Trimble dealer |

**Tradeoff**: Portable pH/EC meters are cost-effective and reliable. NPK test kits have high variance — use only for screening, not model input. Lab tests remain gold standard for N, P, K.

---

## E. MODELING & PROCESSING MECHANICS

### E.1 Recommended Modeling Approach

**Architecture** (matches existing project, extended for location-specificity):

```
Input Features → Feature Engineering → [XGBoost Multi-class Classifier] → Top-3 Crops
                                     → [Per-Crop Yield Regressor (LightGBM)] → kg/ha
                                     → [SHAP Explainer] → Feature Contributions
                                     → [Rule-Based Baseline] → Fallback/Sanity Check
```

- **Classifier**: XGBoost with `objective='multi:softprob'`, `num_class=22` (or filtered to TARGET_CROPS).
- **Yield regressor**: Separate LightGBM per crop (target = district yield kg/ha).
- **Baseline**: Rule-based pH/rainfall/temperature thresholds per crop (existing in `src/models/baseline.py`).
- **Ranking approach**: Post-hoc economic ranking by multiplying probability × expected_yield × market_price.

### E.2 Feature Engineering Specifics

| Feature | Engineering | Region-Specific |
|---------|-------------|-----------------|
| `pH_bin` | Binned: acidic (<5.5), slightly_acidic (5.5–6.5), neutral (6.5–7.5), alkaline (>7.5) | Laterite soils (Kerala/Konkan): acidic bias; alluvial (UP/Bihar): neutral-alkaline |
| `NP_ratio` | N_kg_ha / max(P_kg_ha, 1) | Ideal N:P = 2:1 for cereals |
| `NK_ratio` | N_kg_ha / max(K_kg_ha, 1) | K-deficiency indicator |
| `ndvi_amplitude` | max(NDVI) - min(NDVI) over growing season | Higher for irrigated, seasonal crops |
| `gdd` | Sum of (daily_mean_temp - base_temp) over season | Base: 10°C for wheat, 15°C for rice |
| `moisture_index` | avg_precip_mm - PET (estimated from temp/humidity) | Arid vs humid zone indicator |
| `season_encoded` | One-hot: Kharif, Rabi, Zaid | Determines crop calendar |
| `location_cluster` | K-means cluster ID from (lat, lon) with k=20 | Captures agro-climatic zone |

### E.3 Group/Split Strategy

- **Group by**: `location_id` (5–10 km grid cell, hashed from rounded lat/lon).
- **Split**: GroupKFold (k=5); ensures no spatial leakage.
- **Geo-held-out test**: Hold out 1–2 entire districts for final evaluation.
- **Stratification**: Stratify folds by crop label to maintain class balance.

### E.4 Sample Size Estimate

| Target | Required Samples | Feasibility |
|--------|-----------------|-------------|
| Top-3 accuracy ≥ 0.75 | ~300–500 geo-referenced samples | Achievable with SHC scraping alone |
| Top-3 accuracy ≥ 0.85 | ~800–1,500 samples | Needs SHC + some lab samples |
| Top-3 accuracy ≥ 0.90 | ~2,000+ samples | Needs multi-district data collection |

With the existing Kaggle dataset (2,200 samples, no geo) as pretraining + 500 location-specific SHC records, **Top-3 ≥ 0.80 is feasible** via transfer learning / fine-tuning.

---

## F. PERMISSIONS, ETHICS & CONTACT ACTIONS

### F.1 Permission Status Summary

| Resource | Permission Level | Action Required |
|----------|-----------------|-----------------|
| Kaggle Crop Recommendation | ✅ Public (CC0) | None |
| India Crop Production (data.gov.in) | ✅ Public (GODL) | None |
| ISRIC SoilGrids | ✅ Public (CC-BY 4.0) | Attribution in outputs |
| Open-Meteo | ✅ Public (CC-BY 4.0) | Attribution |
| Sentinel-2 via GEE | ⚠️ Registration | Sign up at earthengine.google.com |
| data.gov.in API | ⚠️ Registration | Free API key at data.gov.in/user/register |
| Soil Health Card Portal | ⚠️ Scraping | No explicit API; use Google scraper (Apache 2.0) |
| Local lab results | ⚠️ Farmer consent | Use consent template below |
| NGO/FPO farmer data | 🔒 MoU required | Contact NGO with permission email |

### F.2 Permission Request Email Template

```
Subject: Data Sharing Request — AI Crop Recommendation Research for {DISTRICT}

Dear [Organization/Lab Director Name],

We are developing an AI-based crop recommendation system to help farmers in
{DISTRICT}, {STATE} select optimal crops based on soil test results and local
climate data.

We would like to request access to anonymized soil test results from your
laboratory/organization for research purposes. Specifically, we need:
- Soil test values: pH, EC, OC, N, P, K (and micronutrients if available)
- Approximate location (village/block level, not individual farms)
- Crop grown and yield (if recorded)

Data use commitments:
1. All data will be used solely for agricultural research
2. No personally identifiable information (PII) will be collected
3. Individual records will not be published; only aggregate insights
4. Compliant with Indian data protection regulations
5. Research outputs will be shared with your organization

We are happy to sign a formal data-sharing agreement (MoU) if required.

Thank you for supporting agricultural innovation.

Sincerely,
[Your Name]
[Organization]
[Contact: email / phone]
```

### F.3 Farmer Consent Form

See existing template at: `docs/consent_template.md` in the project repository. Key elements:
- Simple language (translatable to Hindi/regional language)
- Voluntary participation, right to withdraw
- Data privacy with pseudonymization
- Advisory-only nature of recommendations
- Witness provision for non-literate participants

---

## G. COSTS, TIMELINE & ACCEPTANCE TESTS

### G.1 Cost Breakdown

| Item | Primary Pipeline | Secondary Pipeline | Tertiary Pipeline |
|------|-----------------|-------------------|-------------------|
| Lab soil tests (50 samples × $5) | $250 | $0 | $0 |
| Field kit | $30 | $0 | $0 |
| Field officer time (10 days × $25) | $250 | $0 | $0 |
| API costs | $0 | $0 | $0 |
| Compute (training, GEE, hosting) | $50–$100 | $50 | $20 |
| Developer time (4–8 weeks) | $0 (self) | $0 | $0 |
| **Total estimated cost** | **$580–$630** | **$50** | **$20** |

### G.2 Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| 1. Data Acquisition | Week 1–3 | Geocode location, scrape SHC, collect samples, send to lab |
| 2. ETL & Feature Eng. | Week 3–4 | Harmonize data, fetch weather/NDVI, compute features |
| 3. Model Training | Week 4–5 | Fine-tune XGBoost, train yield regressors, SHAP setup |
| 4. API & Docker | Week 5–6 | Extend FastAPI endpoint, Docker build, health checks |
| 5. Acceptance Tests | Week 6–7 | Run geo-held-out eval, latency benchmark, integration tests |
| 6. Deploy | Week 7–8 | Terraform apply, smoke test, monitoring setup |

### G.3 Acceptance Tests

| Test | Target | Method |
|------|--------|--------|
| Top-3 accuracy (geo-held-out) | ≥ 0.75 | `pytest tests/test_model.py` with held-out district data |
| `/predict` returns top-3 + SHAP | 100% requests | `curl -X POST /predict` with valid payload; verify JSON schema |
| p95 latency | ≤ 300ms | `pytest tests/test_latency.py` (currently achieving 33.9ms) |
| Docker health check | Pass | `docker-compose up -d && curl /health` |
| Data validation (8 checks) | All pass | `python -m src.data.validation` |
| Model size | ≤ 500MB | `ls -lh models/` |

---

## API_SNIPPETS (Programmatic Examples)

### 1. Geocode Indian PIN Code → Lat/Lon
```python
import requests
r = requests.get(f"https://api.postalpincode.in/pincode/{LOCATION}")
data = r.json()[0]["PostOffice"][0]
lat, lon = None, None  # PIN API doesn't return coords; use Google Geocoding or a lookup CSV
# Alternative: https://nominatim.openstreetmap.org/search?postalcode={PIN}&country=IN&format=json
r2 = requests.get(f"https://nominatim.openstreetmap.org/search?postalcode={LOCATION}&country=IN&format=json")
lat, lon = float(r2.json()[0]["lat"]), float(r2.json()[0]["lon"])
```

### 2. Query ISRIC SoilGrids
```python
import requests
url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lat={lat}&lon={lon}&property=phh2o&property=clay&property=soc&property=nitrogen&depth=0-5cm&depth=5-15cm&value=mean"
r = requests.get(url)
soil = r.json()  # {"properties": {"layers": [{"name":"phh2o", "depths":[{"values":{"mean":62}}]}]}}
```

### 3. Fetch Weather from Open-Meteo
```python
import requests
url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2024-01-01&end_date=2024-12-31&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean"
weather = requests.get(url).json()
```

### 4. Sentinel-2 NDVI via Google Earth Engine
```python
import ee
ee.Initialize()
point = ee.Geometry.Point(lon, lat)
s2 = ee.ImageCollection('COPERNICUS/S2_SR').filterBounds(point).filterDate('2024-01-01','2024-12-31').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
def add_ndvi(img):
    return img.addBands(img.normalizedDifference(['B8','B4']).rename('NDVI'))
ndvi_ts = s2.map(add_ndvi).select('NDVI').getRegion(point, 10).getInfo()
```

### 5. Agmarknet Market Prices via data.gov.in
```bash
curl "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key=YOUR_KEY&format=json&limit=10&filters[State.keyword]=Karnataka&filters[Commodity]=Rice"
```

### 6. Call Existing Crop Recommendation API
```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"N":90,"P":42,"K":43,"temperature":20.87,"humidity":82.0,"ph":6.5,"rainfall":202.9}'
```

---

## PERMISSIONS_TEMPLATES

- **Permission email**: See Section F.2 above.
- **Consent form**: See `docs/consent_template.md` (already in project repository).

---

## EXECUTION_PROMPT

```
RUN_LOCATION_PIPELINE LOCATION={LOCATION} PIPELINE=Primary TARGET_CROPS=rice,maize,wheat,soybean,groundnut,sugarcane PRIORITIZE=data_quality RADIUS_KM=50 MODE=full_run
```

> Replace `{LOCATION}` with any Indian PIN code (e.g., `560001`) or coordinates (e.g., `12.9716,77.5946`).
> Change `PIPELINE=Secondary` for zero-cost remote-sensing-only mode.
> Change `MODE=dry_run` to validate data availability without collecting samples.
> Change `MODE=data_only` to stop after ETL without training.
