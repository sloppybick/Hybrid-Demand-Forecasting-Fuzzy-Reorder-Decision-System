"""Fuzzy Decision Engine page: SKU-driven or manually-driven fuzzy inference with explanation."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.cleaner import clean_pipeline
from src.pipeline import compute_recommendation
from src.fuzzy.inference import ReorderFuzzyEngine, FuzzyInputs

st.set_page_config(page_title="Fuzzy Decision Engine", layout="wide")
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"


@st.cache_data(show_spinner=False)
def load():
    sales = pd.read_csv(DATA_DIR / "sales.csv")
    sales = sales.rename(columns={"quantity_sold": "quantity"})[["date", "sku", "quantity"]]
    sales["date"] = pd.to_datetime(sales["date"])
    cleaned = clean_pipeline(sales)
    inventory = pd.read_csv(DATA_DIR / "inventory.csv")
    return cleaned, inventory


@st.cache_resource(show_spinner=False)
def get_engine():
    return ReorderFuzzyEngine()


st.title("🧠 Fuzzy Decision Engine")

mode = st.radio("Mode", ["From SKU data", "Manual inputs"], horizontal=True)
engine = get_engine()

if mode == "From SKU data":
    cleaned, inventory = load()
    sku = st.selectbox("SKU", sorted(inventory["sku"].unique()))
    inv_row = inventory[inventory.sku == sku].iloc[0]
    history = cleaned[cleaned.sku == sku][["date", "quantity"]]
    rec = compute_recommendation(
        sku=sku, history=history, current_stock=float(inv_row["current_stock"]),
        lead_time_days=float(inv_row["lead_time"]),
        minimum_order_quantity=float(inv_row["minimum_order_quantity"]),
        horizon=14, fuzzy_engine=engine,
    )
    decision = rec.fuzzy
    inputs = decision.inputs
    order_qty = rec.order_quantity_fuzzy
else:
    c1, c2 = st.columns(2)
    with c1:
        stock_ratio = st.slider("Stock ratio (current stock / lead-time demand)", 0.0, 3.0, 1.0, 0.05)
        demand_ratio = st.slider("Demand ratio (forecast vs. historical average)", 0.0, 3.0, 1.0, 0.05)
    with c2:
        uncertainty = st.slider("Forecast uncertainty ((upper-lower)/forecast)", 0.0, 1.5, 0.3, 0.05)
        lead_time_days = st.slider("Lead time (days)", 0, 30, 7)
    inputs = FuzzyInputs(stock_ratio, demand_ratio, uncertainty, lead_time_days)
    decision = engine.infer(inputs)
    order_qty = None

st.subheader("Inputs")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Stock ratio", f"{inputs.stock_ratio:.2f}")
c2.metric("Demand ratio", f"{inputs.demand_ratio:.2f}")
c3.metric("Uncertainty", f"{inputs.uncertainty:.2f}")
c4.metric("Lead time (days)", f"{inputs.lead_time_days:.0f}")

st.subheader("Fuzzy Engine Output")
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=decision.urgency_score,
    title={"text": decision.decision},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "#2c3e50"},
        "steps": [
            {"range": [0, 20], "color": "#2ecc71"},
            {"range": [20, 40], "color": "#a3e635"},
            {"range": [40, 60], "color": "#f1c40f"},
            {"range": [60, 80], "color": "#e67e22"},
            {"range": [80, 100], "color": "#e74c3c"},
        ],
    },
))
fig.update_layout(height=320)
st.plotly_chart(fig, use_container_width=True)

if order_qty is not None:
    st.metric("Recommended order quantity", f"{order_qty:.0f} units")

st.subheader("Why?")
st.info(decision.explanation)
