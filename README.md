# Customer-Retention-Intelligence-System
SmartTel AI is an end-to-end customer retention intelligence platform that transforms telecom customer data into actionable business decisions through churn prediction, explainable AI, customer segmentation, revenue impact analysis, and personalized retention recommendations.
# SmartTel Retention Intelligence

A decision-support system for telecom customer retention, built on the IBM Telco Customer Churn dataset.
It answers one question: *given a limited marketing budget, which customers should we target, and how much revenue can we realistically save?*

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python -m pytest -q                                       # 12+ tests, including parity with notebook 06
streamlit run app/app.py                                  # opens http://localhost:8501
```

## How the pieces fit

| Stage | Where | Produces |
|---|---|---|
| Data audit and features | `notebooks/01_eda_and_data_quality.ipynb` | feature store |
| Personas | `notebooks/02_segmentation.ipynb` | persona labels |
| Churn model + SHAP | `notebooks/03_classification_benchmarking.ipynb` | `models/*.joblib`, scored data |
| CLV, ROI, target list | `notebooks/06_clv_roi_simulation.ipynb` | `telco_prioritization_latest.parquet`, `configs/roi_assumptions.json` |
| SHAP reasons for the app | `python scripts/compute_shap_reasons.py` | `telco_shap_reasons_latest.parquet` |
| Dashboard | `app/app.py` (uses `src/roi_simulator.py`) | |

Experimental extensions (`04_survival_analysis`, `05_uplift_modeling`) are **not** used by the dashboard.

## Important caveats

- **Dollar figures depend on placeholder assumptions** (margin, offer cost, discount terms, save rate). Edit them in the app sidebar or in `configs/roi_assumptions.json`.
- The save rate can only be measured by a randomized pilot.
- What-if scenarios are associative, not causal.
- The IBM data is a single snapshot; do not generalize the numbers to other markets.

## Deploy

**Streamlit Community Cloud:** push to GitHub, create an app, set the main file to `app/app.py`, and choose Python 3.12.
Apps can be public, which is fine for this public dataset; use private hosting plus authentication for real customer data.

**Docker:** `docker build -t smarttel . && docker run -p 8501:8501 smarttel`