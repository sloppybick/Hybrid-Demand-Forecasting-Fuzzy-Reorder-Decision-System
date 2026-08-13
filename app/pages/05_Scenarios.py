"""Scenario Simulator: stress-test the fuzzy reorder policy under demand/lead-time shocks."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.cleaner import clean_pipeline
from src.simulation.scenarios import run_all_scenarios

st.set_page_config(page_title="Scenarios", layout="wide")
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"


@st.cache_data(show_spinner=False)
def load():
    sales = pd.read_csv(DATA_DIR / "sales.csv")
    sales = sales.rename(columns={"quantity_sold": "quantity"})[["date", "sku", "quantity"]]
    sales["date"] = pd.to_datetime(sales["date"])
    cleaned = clean_pipeline(sales)
    inventory = pd.read_csv(DATA_DIR / "inventory.csv")
    return cleaned, inventory


st.title("🌪️ Scenario Simulator")
st.caption("Normal vs. demand spike vs. supplier disruption vs. promotional period — same SKU, same policy.")

cleaned, inventory = load()
sku = st.selectbox("SKU", sorted(inventory["sku"].unique()))
inv_row = inventory[inventory.sku == sku].iloc[0]
history = cleaned[cleaned.sku == sku][["date", "quantity"]]

horizon = st.select_slider("Forecast horizon (days)", options=[7, 14, 30], value=14)

with st.spinner("Running scenarios..."):
    df = run_all_scenarios(
        sku=sku, history=history, current_stock=float(inv_row["current_stock"]),
        lead_time_days=float(inv_row["lead_time"]),
        minimum_order_quantity=float(inv_row["minimum_order_quantity"]),
        horizon=horizon,
    )

st.dataframe(df, use_container_width=True, hide_index=True)

fig = px.bar(df, x="scenario", y="reorder_urgency", color="reorder_urgency",
             color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c"], range_color=[0, 100])
fig.update_layout(height=400, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "This demonstrates decision robustness: the same underlying policy reacts proportionally "
    "to demand shocks and supplier disruptions without being manually re-tuned per scenario."
)
