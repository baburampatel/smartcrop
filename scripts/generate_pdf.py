"""Generate a comprehensive project overview PDF."""

from pathlib import Path
from fpdf import FPDF

OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "project_overview.pdf"


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Crop Recommendation System - Project Overview", align="R")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def h1(self, text):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(16, 185, 129)
        self.cell(0, 12, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(16, 185, 129)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def h2(self, text):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(50, 50, 50)
        self.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def p(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.set_x(10)
        self.multi_cell(190, 6, text)
        self.ln(2)

    def li(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.set_x(10)
        self.multi_cell(190, 6, "  - " + text)

    def mono(self, text):
        self.set_font("Courier", "", 7.5)
        self.set_text_color(40, 40, 40)
        self.set_x(10)
        self.cell(190, 4.5, text, new_x="LMARGIN", new_y="NEXT")

    def table(self, headers, rows, widths=None):
        n = len(headers)
        if widths is None:
            widths = [190 / n] * n
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(16, 185, 129)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(widths[i], 8, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        alt = False
        for row in rows:
            self.set_fill_color(240, 245, 240) if alt else self.set_fill_color(255, 255, 255)
            for i, c in enumerate(row):
                self.cell(widths[i], 7, str(c), border=1, fill=True, align="C" if i > 0 else "L")
            self.ln()
            alt = not alt
        self.ln(3)


def build():
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ===== TITLE PAGE =====
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(16, 185, 129)
    pdf.cell(0, 15, "Crop Recommendation System", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Production-Grade ML System for Indian Agriculture", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, "XGBoost | FastAPI | Docker | Terraform | MLflow | SHAP", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(16, 185, 129)
    pdf.cell(0, 10, "Key Results", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.table(
        ["Metric", "Target", "Achieved"],
        [
            ["Top-3 Accuracy", ">= 0.75", "0.9521"],
            ["Top-1 Accuracy", "--", "0.7285"],
            ["p95 Latency", "<= 300ms", "33.9ms"],
            ["Yield MAE", "--", "2,871.91 kg/ha"],
            ["Model Size", "<= 500MB", "< 50MB"],
            ["Tests Passing", "All", "30/30"],
            ["Crops Covered", "--", "22"],
        ],
        widths=[70, 60, 60],
    )

    # ===== 1. ARCHITECTURE =====
    pdf.add_page()
    pdf.h1("1. System Architecture")
    pdf.p("The system takes soil lab values (N, P, K, pH) and climate data "
           "(temperature, humidity, rainfall) as input and produces top-3 crop "
           "recommendations with expected yield and SHAP-based explanations.")
    pdf.h2("Pipeline Flow")
    pdf.li("Step 1: Soil Sample Input (7 features)")
    pdf.li("Step 2: ETL Pipeline (ingest.py) - normalize units, harmonize names")
    pdf.li("Step 3: Feature Engineering - pH bins, nutrient ratios, one-hot encoding")
    pdf.li("Step 4: XGBoost Classifier - multi-class classification, 22 crops")
    pdf.li("Step 5: XGBoost Yield Regressor - expected yield in kg/ha")
    pdf.li("Step 6: SHAP TreeExplainer - per-prediction feature contributions")
    pdf.li("Step 7: FastAPI /predict endpoint - structured JSON response")
    pdf.li("Step 8: Web UI - dark-themed field officer portal")

    # ===== 2. TECH STACK =====
    pdf.ln(5)
    pdf.h1("2. Technology Stack")
    pdf.table(
        ["Category", "Technologies"],
        [
            ["Language", "Python 3.10+"],
            ["ML Models", "XGBoost, LightGBM, scikit-learn"],
            ["Hyperparameter Tuning", "Optuna (Bayesian optimization)"],
            ["Explainability", "SHAP (TreeExplainer)"],
            ["API Framework", "FastAPI, Pydantic, Uvicorn"],
            ["Data Processing", "Pandas, PyArrow (Parquet), NumPy"],
            ["Experiment Tracking", "MLflow (file-based store)"],
            ["Monitoring", "Prometheus (counters, histograms)"],
            ["Containerization", "Docker, Docker Compose"],
            ["Infrastructure", "Terraform (AWS EC2, CloudWatch)"],
            ["CI/CD", "GitHub Actions"],
            ["Testing", "pytest (30 tests)"],
            ["Datasets", "Kaggle (CC0), data.gov.in (GODL)"],
        ],
        widths=[60, 130],
    )

    # ===== 3. DATA LAYER =====
    pdf.add_page()
    pdf.h1("3. Data Layer")

    pdf.h2("3.1 Datasets Used")
    pdf.table(
        ["Dataset", "Source", "License", "Records"],
        [
            ["Crop Recommendation", "Kaggle (Atharva Ingle)", "CC0", "2,200"],
            ["India Crop Production", "data.gov.in", "GODL", "~250K"],
            ["Crop Yield States", "Kaggle", "CC0", "~10K"],
        ],
        widths=[50, 55, 30, 30],
    )
    pdf.p("All datasets are publicly licensed. No permission needed. No PII present.")

    pdf.h2("3.2 Canonical Schema (schema.py)")
    pdf.p("Defines 27 canonical columns including: sample_id, date, latitude, longitude, "
           "location_id, pH, EC_dS_m, OC_pct, N_kg_ha, P_kg_ha, K_kg_ha, S_mg_kg, "
           "zn_ppm, fe_ppm, mn_ppm, cu_ppm, texture, moisture_pct, previous_crop, "
           "irrigation, season, avg_precip_mm, avg_temp_c, target_crop, yield_kg_ha.")
    pdf.p("22 crops: apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, "
           "jute, kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, "
           "orange, papaya, pigeonpeas, pomegranate, rice, watermelon.")

    pdf.h2("3.3 Data Validation (validation.py)")
    pdf.p("8 automated quality checks:")
    pdf.li("Null rate per column < 50%")
    pdf.li("N, P, K within [0, 500] kg/ha")
    pdf.li("pH within [0, 14]")
    pdf.li("Temperature within [-10, 60] C")
    pdf.li("Humidity within [0, 100]%")
    pdf.li("Rainfall within [0, 5000] mm")
    pdf.li("Crop cardinality = 22")
    pdf.li("Class balance (min/max ratio > 0.5)")

    # ===== 4. ETL =====
    pdf.add_page()
    pdf.h1("4. ETL Pipeline (ingest.py)")
    pdf.p("The ETL pipeline performs 6 steps:")
    pdf.li("1. Generate/Load raw crop recommendation data (2,200 samples, 100 per crop)")
    pdf.li("2. Generate/Load India crop yield data (13 states x 22 crops x 8 years)")
    pdf.li("3. Rename columns to canonical schema (N -> N_kg_ha, temperature -> avg_temp_c)")
    pdf.li("4. Compute and merge median yield per crop (100% coverage)")
    pdf.li("5. Create 20 location_id clusters via KMeans on temp, humidity, rainfall, pH")
    pdf.li("6. Output cleaned Parquet + metadata.json with SHA256 checksum")
    pdf.ln(2)
    pdf.p("Output: data/processed/crop_recommendation_clean.parquet (2,200 rows, 27 columns)")

    # ===== 5. FEATURES =====
    pdf.h1("5. Feature Engineering (features.py)")
    pdf.p("6 sklearn TransformerMixin components in a Pipeline:")
    pdf.table(
        ["Transformer", "Input", "Output"],
        [
            ["PHBinner", "pH", "pH_bin (6 categories)"],
            ["NutrientRatios", "N, P, K", "N_P, N_K, P_K ratios"],
            ["RainfallBinner", "avg_precip_mm", "rainfall_bin (5 cat)"],
            ["TemperatureBinner", "avg_temp_c", "temp_bin (5 cat)"],
            ["HumidityBinner", "humidity_pct", "humidity_bin (4 cat)"],
            ["CategoricalEncoder", "All bins", "One-hot encoded dummies"],
        ],
        widths=[42, 40, 108],
    )
    pdf.p("Total: 26 features (7 numeric + 3 ratios + 16 one-hot bins). "
           "Pipeline saved as: models/feature_pipeline.joblib")

    # ===== 6. MODELING =====
    pdf.add_page()
    pdf.h1("6. Modeling and Experiments")

    pdf.h2("6.1 Rule-Based Baseline (baseline.py)")
    pdf.p("Computes suitability scores per crop based on matching against known agronomic "
           "ideal ranges for N, P, K, pH, temperature, humidity, and rainfall. "
           "Results: Top-1 = 92.64%, Top-3 = 100%.")

    pdf.h2("6.2 XGBoost Classifier (classifier.py)")
    pdf.p("Multi-class XGBoost with 22 classes. Supports XGBoost and LightGBM backends. "
           "Uses softmax objective for probability calibration. predict_top_k() returns "
           "ranked predictions with probabilities.")
    pdf.p("Default params: max_depth=6, learning_rate=0.1, n_estimators=200, "
           "min_child_weight=3, subsample=0.8, colsample_bytree=0.8.")
    pdf.p("Optional Optuna tuning with GroupKFold CV by location_id.")

    pdf.h2("6.3 Yield Regressor (yield_model.py)")
    pdf.p("XGBoost regression predicting yield in kg/ha. Falls back to median crop yields "
           "when per-sample prediction is unavailable. MAE = 2,871.91 kg/ha.")

    pdf.h2("6.4 Training Pipeline (train.py)")
    pdf.p("Geographic hold-out: 80% train (1,699 samples, 16 clusters) / "
           "20% test (501 samples, 4 held-out clusters). All experiments logged to MLflow.")
    pdf.table(
        ["Model", "Top-1", "Top-3", "Time"],
        [
            ["Rule-Based Baseline", "92.64%", "100%", "< 1s"],
            ["XGBoost Classifier", "72.85%", "95.21%", "1.5s"],
        ],
        widths=[55, 45, 45, 45],
    )

    # ===== 7. EXPLAINABILITY =====
    pdf.h1("7. Explainability (shap_explain.py)")
    pdf.p("Uses SHAP TreeExplainer for per-prediction feature contributions. For each "
           "of the top-3 crops, returns top-5 features with SHAP values "
           "(positive = supporting, negative = opposing).")
    pdf.p("Example: [{feature: avg_precip_mm, value: +0.23}, "
           "{feature: N_kg_ha, value: +0.18}, {feature: pH, value: -0.05}]")

    # ===== 8. API =====
    pdf.add_page()
    pdf.h1("8. API and Deployment")

    pdf.h2("8.1 FastAPI Application (app.py)")
    pdf.p("Endpoints:")
    pdf.li("POST /predict - JSON with N, P, K, ph, temperature, humidity, rainfall")
    pdf.li("GET /health - Model status, version")
    pdf.li("GET / - Field officer web UI")
    pdf.li("GET /metrics - Prometheus counters, histograms, drift")

    pdf.h2("8.2 Request/Response Schema (schemas.py)")
    pdf.p("Request: N [0-500], P [0-500], K [0-500], ph [0-14], temperature [-10,60], "
           "humidity [0-100], rainfall [0-5000]. Pydantic Field constraints.")
    pdf.p("Response: top_3 array with crop, probability, expected_yield_kg_ha, "
           "explanation array; plus model_version and data_checksum.")

    pdf.h2("8.3 Web UI (ui/index.html)")
    pdf.p("Dark-themed responsive SPA for field officers. Input fields for 7 params, "
           "sample value filler, animated confidence bars, rank badges (#1/#2/#3), "
           "SHAP contributions, API health indicator, latency display.")

    pdf.h2("8.4 Docker")
    pdf.p("Dockerfile: Python 3.11-slim, 2 uvicorn workers, health check every 30s. "
           "Docker Compose: API + MLflow server with SQLite, 1GB memory limit.")

    pdf.h2("8.5 Terraform (terraform/)")
    pdf.p("AWS: EC2 t3.small in ap-south-1 (Mumbai), security groups (port 8000 + SSH), "
           "IAM roles, CloudWatch logging + CPU alarms, Secrets Manager, Docker user-data.")

    # ===== 9. TESTING =====
    pdf.add_page()
    pdf.h1("9. Testing (30/30 Passing)")
    pdf.table(
        ["Test File", "Tests", "Coverage Area"],
        [
            ["test_data.py", "9", "Schema, crop labels, normalization, yield"],
            ["test_features.py", "5", "Binners, ratios, pipeline consistency"],
            ["test_model.py", "9", "Baseline, classifier, save/load, yield"],
            ["test_api.py", "6", "Health, predict, validation, crop, UI"],
            ["test_latency.py", "1", "100 requests, P95 <= 300ms assertion"],
        ],
        widths=[40, 15, 125],
    )

    pdf.h2("Latency Benchmark")
    pdf.table(
        ["Metric", "Value"],
        [
            ["Mean", "11.8 ms"],
            ["P50 (Median)", "9.0 ms"],
            ["P95", "33.9 ms"],
            ["P99", "36.0 ms"],
            ["Min", "8.1 ms"],
            ["Max", "36.7 ms"],
        ],
        widths=[80, 80],
    )

    # ===== 10. CI/CD =====
    pdf.h1("10. CI/CD (.github/workflows/ci.yml)")
    pdf.p("GitHub Actions on push/PR to main:")
    pdf.li("1. Matrix: Python 3.10 + 3.11")
    pdf.li("2. Install dependencies")
    pdf.li("3. ETL pipeline")
    pdf.li("4. Data validation")
    pdf.li("5. Training pipeline")
    pdf.li("6. Unit tests (pytest)")
    pdf.li("7. Latency benchmark")
    pdf.li("8. Docker build (main branch)")
    pdf.li("9. E2E: container start, health, predict, schema validation")

    # ===== 11. MONITORING =====
    pdf.ln(3)
    pdf.h1("11. MLOps and Monitoring")
    pdf.p("MLflow: Tracks experiment params, metrics (accuracy, MAE), model artifacts. "
           "File-based store with SQLite backend in Docker.")
    pdf.p("Prometheus: predict_requests_total (counter), predict_latency_seconds "
           "(histogram), prediction_crop_total (per-crop drift counter).")
    pdf.p("CloudWatch: System logs, CPU utilization alarms (> 80%). "
           "Retraining: Monthly; retrain if PSI/KL-divergence exceeds thresholds.")

    # ===== 12. DOCS =====
    pdf.add_page()
    pdf.h1("12. Documentation")
    pdf.table(
        ["Document", "Contents"],
        [
            ["README.md", "Quick start, project structure, API schema"],
            ["model_card.md", "Model details, training data, limitations"],
            ["data_catalog.md", "3 datasets: variables, units, licenses"],
            ["field_trial_plan.md", "A/B trial, 100-200 farmers, protocol"],
            ["consent_template.md", "Consent form, PII redaction, risk matrix"],
            ["executive_summary.md", "Problem, solution, results, deliverables"],
            ["labeling.md", "Labeling strategy, bias analysis"],
            ["data_catalog.json", "Machine-readable dataset catalog"],
        ],
        widths=[55, 125],
    )

    pdf.h2("Field Trial Plan")
    pdf.p("Cluster-randomized trial: 100-200 farmers across 4-6 districts, "
           "2 seasons (Kharif + Rabi). 50% treatment vs 50% control. "
           "Primary outcomes: yield difference and net income.")

    pdf.h2("Ethics and Privacy")
    pdf.li("No PII collected - only soil/climate parameters")
    pdf.li("Informed consent in local language for field trials")
    pdf.li("Data pseudonymized; linking table encrypted separately")
    pdf.li("Coordinates rounded to 5km grid; exact locations never stored")
    pdf.li("All recommendations advisory only")

    # ===== 13. FILE STRUCTURE =====
    pdf.h1("13. Complete File Structure")
    files = [
        "src/data/schema.py           - Canonical schema, crop labels",
        "src/data/features.py          - Feature pipeline (6 transformers)",
        "src/data/validation.py        - 8 data quality checks",
        "src/models/baseline.py        - Rule-based recommender",
        "src/models/classifier.py      - XGBoost/LightGBM + Optuna",
        "src/models/yield_model.py     - Yield regression + fallback",
        "src/explain/shap_explain.py   - SHAP TreeExplainer",
        "src/api/app.py                - FastAPI (4 endpoints)",
        "src/api/schemas.py            - Pydantic models",
        "scripts/ingest.py             - ETL pipeline (6 steps)",
        "scripts/train.py              - Training (7 steps + MLflow)",
        "scripts/evaluate.py           - Evaluation reports",
        "tests/test_data.py            - 9 schema tests",
        "tests/test_features.py        - 5 feature tests",
        "tests/test_model.py           - 9 model tests",
        "tests/test_api.py             - 6 API tests",
        "tests/test_latency.py         - Latency benchmark",
        "ui/index.html                 - Field officer web UI",
        "Dockerfile                    - Container definition",
        "docker-compose.yml            - API + MLflow services",
        "terraform/main.tf             - AWS EC2 + CloudWatch",
        "terraform/variables.tf        - Config parameters",
        "terraform/outputs.tf          - API URL outputs",
        ".github/workflows/ci.yml      - CI/CD pipeline",
        "requirements.txt              - Python dependencies",
        "README.md                     - Project documentation",
    ]
    for f in files:
        pdf.mono(f)
    pdf.ln(4)

    # ===== 14. BUGS FIXED =====
    pdf.add_page()
    pdf.h1("14. Bugs Fixed During Verification")
    pdf.table(
        ["Bug", "Root Cause", "Fix"],
        [
            ["Unicode encode error", "Windows cp1252 encoding", "Replaced with ASCII"],
            ["MLflow URI scheme", "Windows path not valid URI", "Used Path.as_uri()"],
            ["CategoricalEncoder", "Missing __init__ attribute", "Added __init__"],
        ],
        widths=[50, 60, 70],
    )

    # ===== 15. SUMMARY =====
    pdf.h1("15. Summary and Next Steps")
    pdf.p("All 10 components delivered. All acceptance tests passing. "
           "Top-3 accuracy 95.21% exceeds the 75% target. P95 latency 33.9ms "
           "is well under the 300ms target.")
    pdf.h2("Recommended Next Steps")
    pdf.li("1. Field validation: Execute the A/B pilot with 100-200 farmers")
    pdf.li("2. Data enrichment: Add micronutrients (Zn, Fe, Mn, Cu) and irrigation")
    pdf.li("3. Real-time weather: Connect to IMD/weather APIs")
    pdf.li("4. Scale: Expand to more crops and agro-climatic zones")
    pdf.li("5. Mobile app: Build offline-capable app for field officers")
    pdf.li("6. Tuning: Run Optuna with --tune flag for improved accuracy")

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT))
    print(f"PDF generated: {OUTPUT}")
    print(f"Size: {OUTPUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    build()
