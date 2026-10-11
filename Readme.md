# E-Commerce Anomaly Agent

An end-to-end e-commerce analytics project built around the **Olist Brazilian E-Commerce Public Dataset**. The project combines sales analytics, daily anomaly detection, anomaly investigation, customer segmentation, and an exploratory churn-model evaluation. A Streamlit dashboard provides a starting point for reviewing flagged sales days and investigating category and seller contributions.

> **Project status:** Core analysis code and selected processed outputs are in the repository. The dashboard is an early working interface, not a production monitoring service. Notebooks 04–07 currently exist as empty placeholders; the corresponding analysis outputs were generated separately and are included where listed below. The project does not yet use a large language model to generate explanations, and anomaly flags are not proof of fraud or business causation.

## Contents

- [What it does](#what-it-does)
- [Dataset](#dataset)
- [Repository structure](#repository-structure)
- [Key analytical results](#key-analytical-results)
- [Methods and caveats](#methods-and-caveats)
- [Setup](#setup)
- [Running the project](#running-the-project)
- [Generated outputs](#generated-outputs)
- [Limitations and next steps](#limitations-and-next-steps)

## What it does

- **Data quality and preparation:** inspect the source tables, standardize date and numeric fields, and prepare cleaned datasets.
- **Sales analytics:** calculate daily and monthly product-revenue and order metrics.
- **Anomaly detection:** use Isolation Forest to flag unusual daily sales activity and rank candidate dates for review.
- **Anomaly investigation:** compare an anomaly date with a preceding baseline and examine category- and seller-level contributions.
- **Customer analytics:** build Recency, Frequency, and Monetary (RFM) segments.
- **Churn-model experiment:** evaluate a simple logistic-regression baseline for a retrospective 90-day inactivity label.
- **Dashboard:** inspect prioritized anomaly records, review available investigation summaries, and explore monthly revenue in Streamlit.

## Dataset

The project uses the [Olist Brazilian E-Commerce Public Dataset on Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). It contains related tables for orders, order items, payments, customers, products, sellers, reviews, product-category translations, and geolocation.

The raw data is **not committed** to this repository. Download it from the dataset source and place the CSV files in `data/raw/`. Keep the original filenames, such as `olist_orders_dataset.csv`, `olist_order_items_dataset.csv`, `olist_order_payments_dataset.csv`, `olist_customers_dataset.csv`, `olist_products_dataset.csv`, `olist_sellers_dataset.csv`, `olist_order_reviews_dataset.csv`, `product_category_name_translation.csv`, and `olist_geolocation_dataset.csv`.

The project’s `.gitignore` excludes `data/raw/` and selected large derived CSVs to keep the Git repository smaller. The data remains your responsibility to download locally before rerunning the full pipeline.

## Repository structure

```text
Anomaly-Agent-Project/
├── app.py
├── main.py
├── requirements.txt
├── Readme.md
├── data/
│   └── processed/       # selected cleaned tables and analysis outputs
├── notebooks/
│   ├── 01_raw_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_sales_revenue_analysis.ipynb
│   ├── 04_anomaly_detection.ipynb   # currently empty placeholder
│   ├── 05_customer_rfm.ipynb        # currently empty placeholder
│   ├── 06_churn_prediction.ipynb    # currently empty placeholder
│   └── 07_business_insights.ipynb   # currently empty placeholder
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_loader.py
    ├── cleaning.py
    └── feature_engineering.py
```

### Main code modules

- `src/config.py` — defines project and data-directory paths.
- `src/data_loader.py` — provides CSV-loading functionality.
- `src/cleaning.py` — includes helpers for datetime and numeric conversion.
- `src/feature_engineering.py` — contains order-level feature creation, analytical dataset construction, and category/seller anomaly-investigation utilities.
- `app.py` — Streamlit dashboard for reviewing precomputed outputs and running an on-demand investigation.
- `main.py` — basic project-path/environment check; it does not run the full analytics pipeline.

## Key analytical results

The following results come from the project analysis recorded during development:

| Measure | Result |
|---|---:|
| Orders in the analytical order-level dataset | 99,441 |
| Unique order IDs in that dataset | 99,441 |
| Duplicate order IDs in that dataset | 0 |
| Product revenue (item prices) | R$13,591,643.70 |
| Freight total | R$2,251,909.54 |
| Payment total | R$16,008,872.12 |
| Average product revenue per order | R$136.68 |
| Daily observations used for the anomaly analysis | 634 |
| Days flagged by the Isolation Forest run | 19 (3% contamination setting) |
| Unique customers used in RFM analysis | 96,096 |
| Repeat customers identified in RFM analysis | 2,997 |

The order-level product-revenue and freight totals reconciled to the item-level sums in the validation performed during development. Payment totals represent a separate measure and should not be interpreted as identical to product revenue plus freight in every row-level context.

### Example: 24 November 2017

The analysis highlighted **24 November 2017** as a high-sales date. Against a 28-day preceding baseline, several categories showed large positive differences in product revenue:

| Category | Revenue on target date | Baseline average per day | Difference |
|---|---:|---:|---:|
| bed_bath_table | R$17,875.03 | R$1,869.90 | +R$16,005.13 |
| watches_gifts | R$17,667.02 | R$2,498.95 | +R$15,168.08 |
| furniture_decor | R$11,729.21 | R$1,306.93 | +R$10,422.28 |
| health_beauty | R$11,779.52 | R$1,768.95 | +R$10,010.57 |
| computers_accessories | R$10,924.12 | R$1,796.08 | +R$9,128.04 |

This date is consistent with the timing of Black Friday, but the date/category association alone does not establish the cause of the increase.

## Methods and caveats

### Daily sales anomaly detection

An Isolation Forest run used daily order counts and product revenue, with a contamination parameter of `0.03`. The analysis excluded the first and last partial months from the main daily comparison and evaluated 634 daily observations. Flagged dates are **candidates for investigation**, not confirmed incidents.

The prioritization logic also compares daily order/revenue measures with shifted rolling 28-day baselines (requiring a minimum history). A day with unusual activity can reflect promotions, seasonality, data issues, or other causes; contextual investigation is required.

### Category and seller investigation

For a selected date, the investigation utilities compare category and seller product revenue against a preceding baseline. These contribution tables help narrow down where a change occurred. They do not prove that any particular seller or category caused an anomaly.

### RFM segmentation

Customer RFM features summarize recency, purchase frequency, and monetary value using the available historical orders. Segment labels are analytical groupings, not guarantees of future customer behavior.

### Churn-model experiment

The current logistic-regression experiment uses frequency and monetary features to predict a retrospective inactivity label based on days since the last observed purchase. The recorded ROC-AUC was approximately **0.498**, which is near random discrimination. Treat this as an exploratory baseline, **not a reliable churn predictor**. The target construction, temporal validation, feature set, and class handling need further work before any business use.

## Setup

Use Python 3.11 or 3.12 as a practical starting point for the listed libraries. The notebooks were last recorded with a Python 3.14 environment, but the full environment has not been validated from a clean installation.

### 1. Clone the repository

```bash
git clone https://github.com/Mrityunjai-hue/Anomaly-Agent-Project.git
cd Anomaly-Agent-Project
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install streamlit jupyter
```

Streamlit and Jupyter are installed separately because they are currently used by the app/notebooks but are not listed in the repository’s present `requirements.txt`.

### 4. Download the dataset

Download the Olist CSVs from the [Kaggle dataset page](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), then place them in `data/raw/`. The repository’s existing processed outputs are useful for reviewing results, but the raw inputs are needed to rebuild the complete cleaning and feature-generation workflow.

## Running the project

### Check project paths

```bash
python main.py
```

This prints the configured project and data directories. It does **not** run all analyses.

### Open the notebooks

```bash
jupyter notebook
```

Start with:

1. `notebooks/01_raw_data_exploration.ipynb`
2. `notebooks/02_data_cleaning.ipynb`
3. `notebooks/03_sales_revenue_analysis.ipynb`

Notebooks 04–07 are currently empty in the public repository and need to be populated before they can be used as the documented execution sequence for anomaly detection, RFM, churn modelling, and business insights.

### Launch the dashboard

```bash
streamlit run app.py
```

The dashboard reads precomputed CSVs from `data/processed/`. It expects, at minimum, `prioritized_anomalies.csv`, `ranked_anomaly_investigation_report.csv`, `monthly_sales_metrics.csv`, `orders_cleaned.csv`, `order_items_cleaned.csv`, `products_cleaned.csv`, `sellers_cleaned.csv`, and `category_translation_cleaned.csv`. If any required file is missing, the dashboard will display an error or the relevant section will not be usable. The dashboard does not regenerate every output automatically.

## Generated outputs

Selected tracked files in `data/processed/` include:

- `daily_sales_metrics.csv` and `monthly_sales_metrics.csv` — daily/monthly sales aggregates.
- `daily_sales_anomaly_results.csv`, `prioritized_anomalies.csv`, `anomaly_report.csv`, and `anomaly_summary.csv` — anomaly flags and summaries.
- `ranked_anomaly_investigation_report.csv`, `anomaly_investigation_summary.csv`, `investigation_findings.csv`, and `sales_anomaly_investigation.csv` — selected investigation outputs.
- `anomaly_category_contributions.csv`, `anomaly_seller_contributions.csv`, `investigated_category_contributions.csv`, and `investigated_seller_contributions.csv` — contribution analyses.
- `customer_rfm_segments.csv` — customer-level RFM segmentation.
- `churn_model_evaluation.csv` and `churn_test_predictions.csv` — exploratory churn-model outputs.
- `delivery_review_analysis.csv` and `business_insights.csv` — supporting descriptive analyses.

Some larger processed files are intentionally ignored by Git. The list above is not a guarantee that every output can be regenerated with a single command in the current repository state.

## Limitations and next steps

- Populate notebooks 04–07 and make their outputs reproducible from the tracked code.
- Add `streamlit` and `jupyter` to `requirements.txt` so setup is reproducible with one install command.
- Add automated tests and data/schema validation.
- Improve the churn experiment with a clearly defined prediction horizon, time-based validation, stronger features, and suitable baseline metrics.
- Consider zero-filling dates with no orders before calculating rolling daily baselines.
- Add documented orchestration so a fresh clone can build all outputs from raw data.
- Develop an explainable investigation layer that distinguishes measured contributions from hypotheses; any future LLM-generated narrative should be grounded in computed evidence and clearly marked as interpretation.

## Acknowledgements

This project uses the Olist Brazilian E-Commerce Public Dataset. Refer to the dataset page and its terms for attribution and use conditions.
