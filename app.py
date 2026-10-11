
import pandas as pd
import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

st.set_page_config(
    page_title="Olist Anomaly Agent",
    page_icon="📊",
    layout="wide"
)

st.title("Olist Anomaly Agent")
st.caption("E-commerce monitoring and anomaly investigation")

required_files = {
    "anomalies": "prioritized_anomalies.csv",
    "investigations": "ranked_anomaly_investigation_report.csv",
    "monthly_sales": "monthly_sales_metrics.csv",
}

missing_files = [
    filename
    for filename in required_files.values()
    if not (PROCESSED_DIR / filename).exists()
]

if missing_files:
    st.error("Missing required output files:")
    for filename in missing_files:
        st.write(f"- {filename}")
    st.info(
        "Generate the missing outputs in Notebook 07, "
        "then reload this dashboard."
    )
    st.stop()

anomalies = pd.read_csv(
    PROCESSED_DIR / required_files["anomalies"],
    parse_dates=["purchase_date"]
)

investigations = pd.read_csv(
    PROCESSED_DIR / required_files["investigations"],
    parse_dates=["anomaly_date"]
)

monthly_sales = pd.read_csv(
    PROCESSED_DIR / required_files["monthly_sales"]
)

# KPI summary
c1, c2, c3 = st.columns(3)

c1.metric("Flagged days", len(anomalies))
c2.metric(
    "High severity",
    int((anomalies["severity"] == "High").sum())
)
c3.metric(
    "Investigations available",
    len(investigations)
)

st.subheader("Detected anomalies")

display_columns = [
    column
    for column in [
        "purchase_date",
        "total_orders",
        "product_revenue",
        "estimated_revenue_deviation",
        "severity",
        "severity_score"
    ]
    if column in anomalies.columns
]

st.dataframe(
    anomalies[display_columns],
    use_container_width=True,
    hide_index=True
)

st.subheader("Investigate an anomaly")

if not investigations.empty:
    available_dates = investigations["anomaly_date"].dropna()

    if not available_dates.empty:
        selected_date = st.selectbox(
            "Choose an investigated date",
            sorted(available_dates.dt.strftime("%Y-%m-%d").unique())
        )

        selected = investigations[
            investigations["anomaly_date"].dt.strftime("%Y-%m-%d")
            == selected_date
        ]

        st.dataframe(
            selected,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No valid investigation dates are available.")
else:
    st.info("No investigation reports are available yet.")

st.subheader("Monthly product revenue")

if {"purchase_month", "product_revenue"}.issubset(
    monthly_sales.columns
):
    chart_data = monthly_sales.copy()
    chart_data["purchase_month"] = pd.to_datetime(
        chart_data["purchase_month"], errors="coerce"
    )
    chart_data = chart_data.dropna(subset=["purchase_month"])

    st.line_chart(
        chart_data.set_index("purchase_month")["product_revenue"]
    )

st.caption(
    "Anomaly flags indicate unusual patterns, not confirmed fraud "
    "or proven business causes. Review the supporting evidence "
    "before taking action."
)

import sys
import importlib

# Make the project root importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.feature_engineering as fe
importlib.reload(fe)

st.subheader("Investigate any detected anomaly")

orders = pd.read_csv(
    PROCESSED_DIR / "orders_cleaned.csv",
    parse_dates=[
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
)

order_items = pd.read_csv(
    PROCESSED_DIR / "order_items_cleaned.csv"
)
products = pd.read_csv(
    PROCESSED_DIR / "products_cleaned.csv"
)
sellers = pd.read_csv(
    PROCESSED_DIR / "sellers_cleaned.csv"
)
translations = pd.read_csv(
    PROCESSED_DIR / "category_translation_cleaned.csv"
)

if not anomalies.empty:
    date_options = (
        anomalies["purchase_date"]
        .dropna()
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )

    selected_date = st.selectbox(
        "Select a flagged date",
        date_options,
        key="investigation_date"
    )

    if st.button("Run investigation"):
        try:
            result = fe.investigate_sales_anomaly(
                order_items=order_items,
                orders=orders,
                products=products,
                category_translation=translations,
                sellers=sellers,
                target_date=selected_date,
                baseline_days=28
            )

            st.markdown("### Investigation summary")
            st.json(result["summary"])

            st.markdown("### Top contributing categories")
            st.dataframe(
                result["category_contributions"].head(10),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### Top contributing sellers")
            st.dataframe(
                result["seller_contributions"].head(10),
                use_container_width=True,
                hide_index=True
            )

        except (ValueError, KeyError) as error:
            st.warning(f"Investigation could not be completed: {error}")
