"""
Home page: inventory health overview + reorder queue.
Run with: streamlit run app/Home.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cleaner import clean_pipeline
from src.pipeline import compute_recommendation
from src.fuzzy.inference import ReorderFuzzyEngine

st.set_page_config(page_title="Demand + Fuzzy Reorder", layout="wide", page_icon="📦")

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


@st.cache_data(show_spinner=False)
def load_data():
    sales = pd.read_csv(DATA_DIR / "sales.csv")
    sales = sales.rename(columns={"quantity_sold": "quantity"})[["date", "sku", "quantity"]]
    sales["date"] = pd.to_datetime(sales["date"])
    cleaned = clean_pipeline(sales)
    inventory = pd.read_csv(DATA_DIR / "inventory.csv")
    return cleaned, inventory


@st.cache_data(show_spinner=True)
def compute_all_recommendations(_cleaned: pd.DataFrame, _inventory: pd.DataFrame, horizon: int, service_level: float):
    engine = ReorderFuzzyEngine()
    rows = []
    for _, inv_row in _inventory.iterrows():
        sku = inv_row["sku"]
        history = _cleaned[_cleaned.sku == sku][["date", "quantity"]]
        if history.empty:
            continue
        rec = compute_recommendation(
            sku=sku,
            history=history,
            current_stock=inv_row["current_stock"],
            lead_time_days=inv_row["lead_time"],
            minimum_order_quantity=inv_row["minimum_order_quantity"],
            horizon=horizon,
            service_level=service_level,
            fuzzy_engine=engine,
        )
        row = rec.to_row()
        row["lead_time"] = inv_row["lead_time"]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    st.title("📦 Demand Forecasting + Fuzzy Reorder Advisor")
    st.caption(
        "Prophet predicts demand. Fuzzy logic decides what to do about it. "
        "Companion project to OptiRoute (route optimization) and the RL Dynamic Pricing Agent."
    )

    with st.sidebar:
        st.header("Settings")
        horizon = st.slider("Forecast horizon (days)", 7, 60, 14, step=7)
        service_level = st.select_slider(
            "Service level (classical policy)",
            options=[0.90, 0.95, 0.975, 0.98, 0.99],
            value=0.95,
            format_func=lambda x: f"{x:.1%}",
        )
        st.markdown("---")
        st.markdown(
            "**Pages:** Data Explorer · Forecasting · Inventory Intelligence · "
            "Fuzzy Decision Engine · Scenarios · Model Evaluation"
        )

    cleaned, inventory = load_data()
    recs = compute_all_recommendations(cleaned, inventory, horizon, service_level)

    st.subheader("Inventory Health")
    c1, c2, c3, c4 = st.columns(4)
    n_skus = len(recs)
    n_critical = int((recs["reorder_urgency_fuzzy"] >= 80).sum())
    n_reorder = int((recs["reorder_urgency_fuzzy"] >= 40).sum())
    avg_urgency = recs["reorder_urgency_fuzzy"].mean() if n_skus else 0
    c1.metric("Total SKUs", n_skus)
    c2.metric("Critical SKUs (urgency ≥ 80)", n_critical)
    c3.metric("SKUs Needing Reorder (≥ 40)", n_reorder)
    c4.metric("Avg. Reorder Urgency", f"{avg_urgency:.1f} / 100")

    st.subheader("Reorder Queue")
    display_cols = [
        "sku", "reorder_urgency_fuzzy", "decision_fuzzy", "current_stock",
        "forecast_demand", "order_qty_fuzzy", "decision_classical", "order_qty_classical",
    ]
    sorted_recs = recs.sort_values("reorder_urgency_fuzzy", ascending=False)

    def highlight_urgency(val):
        if val >= 80:
            return "background-color: #ffcccc"
        if val >= 60:
            return "background-color: #ffe0b3"
        if val >= 40:
            return "background-color: #fff5cc"
        return ""

    styled = sorted_recs[display_cols].style.map(
        highlight_urgency, subset=["reorder_urgency_fuzzy"]
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.subheader("Reorder Urgency by SKU")
    fig = px.bar(
        sorted_recs, x="sku", y="reorder_urgency_fuzzy", color="reorder_urgency_fuzzy",
        color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c"],
        range_color=[0, 100], labels={"reorder_urgency_fuzzy": "Urgency"},
    )
    fig.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Use the pages in the sidebar to drill into a single SKU's forecast, "
        "see the fuzzy engine's reasoning, or stress-test decisions under demand/lead-time scenarios."
    )


if __name__ == "__main__":
    main()
