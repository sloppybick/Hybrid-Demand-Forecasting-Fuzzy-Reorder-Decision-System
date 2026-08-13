"""Inventory Intelligence: classical (s, Q) reorder-point policy details per SKU."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.cleaner import clean_pipeline
from src.pipeline import compute_recommendation

st.set_page_config(page_title="Inventory Intelligence", layout="wide")
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"


@st.cache_data(show_spinner=False)
def load():
    sales = pd.read_csv(DATA_DIR / "sales.csv")
    sales = sales.rename(columns={"quantity_sold": "quantity"})[["date", "sku", "quantity"]]
    sales["date"] = pd.to_datetime(sales["date"])
    cleaned = clean_pipeline(sales)
    inventory = pd.read_csv(DATA_DIR / "inventory.csv")
    return cleaned, inventory


st.title("🏭 Inventory Intelligence")
st.caption("The classical (s, Q) reorder-point baseline — what the fuzzy engine is competing against.")

cleaned, inventory = load()
sku = st.selectbox("SKU", sorted(inventory["sku"].unique()))
inv_row = inventory[inventory.sku == sku].iloc[0]

col_a, col_b, col_c = st.columns(3)
with col_a:
    current_stock = st.number_input("Current stock", value=float(inv_row["current_stock"]), min_value=0.0)
with col_b:
    lead_time = st.number_input("Lead time (days)", value=float(inv_row["lead_time"]), min_value=1.0)
with col_c:
    service_level = st.select_slider(
        "Service level", options=[0.90, 0.95, 0.975, 0.98, 0.99], value=0.95,
        format_func=lambda x: f"{x:.1%}",
    )

history = cleaned[cleaned.sku == sku][["date", "quantity"]]
rec = compute_recommendation(
    sku=sku, history=history, current_stock=current_stock, lead_time_days=lead_time,
    minimum_order_quantity=float(inv_row["minimum_order_quantity"]), horizon=30, service_level=service_level,
)

st.subheader("Reorder Point Calculation")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Expected lead-time demand", f"{rec.classical.lead_time_demand:.1f}")
c2.metric("Safety stock", f"{rec.classical.safety_stock:.1f}")
c3.metric("Reorder point (ROP)", f"{rec.classical.reorder_point:.1f}")
c4.metric("Current stock", f"{current_stock:.0f}")

if rec.classical.should_reorder:
    st.error(
        f"⚠️ Current stock ({current_stock:.0f}) is at or below the reorder point "
        f"({rec.classical.reorder_point:.1f}) — REORDER"
    )
    st.metric("Recommended order quantity (classical)", f"{rec.order_quantity_classical:.0f} units")
else:
    st.success(
        f"✅ Current stock ({current_stock:.0f}) is above the reorder point "
        f"({rec.classical.reorder_point:.1f}) — HOLD"
    )

st.markdown("---")
st.subheader("Formula reference")
st.latex(r"ROP = D_{LT} + SS \qquad SS = z \cdot \sigma_{daily}\sqrt{L}")
