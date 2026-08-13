"""Data Explorer: raw vs cleaned demand, EDA statistics, seasonality."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.cleaner import clean_pipeline
from src.data.aggregator import (
    demand_statistics, rolling_features, weekly_seasonality_index, linear_trend_slope,
)

st.set_page_config(page_title="Data Explorer", layout="wide")
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"


@st.cache_data(show_spinner=False)
def load():
    sales = pd.read_csv(DATA_DIR / "sales.csv")
    sales = sales.rename(columns={"quantity_sold": "quantity"})[["date", "sku", "quantity"]]
    sales["date"] = pd.to_datetime(sales["date"])
    cleaned = clean_pipeline(sales)
    return sales, cleaned


st.title("🔍 Data Explorer")

raw, cleaned = load()

sku = st.selectbox("SKU", sorted(cleaned["sku"].unique()))
g = cleaned[cleaned.sku == sku].sort_values("date")
g_roll = rolling_features(g, window=7)

st.subheader(f"Historical Demand — {sku}")
fig = px.line(g_roll, x="date", y=["quantity", "rolling_mean_7d"], labels={"value": "units", "variable": ""})
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Rolling Volatility (7-day std)")
    fig2 = px.line(g_roll, x="date", y="rolling_std_7d")
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("Demand Distribution")
    fig3 = px.histogram(g, x="quantity", nbins=30)
    st.plotly_chart(fig3, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.subheader("Weekly Seasonality Index")
    st.caption("1.0 = average day; >1.0 = above-average demand")
    idx = weekly_seasonality_index(g).reset_index()
    idx.columns = ["day_of_week", "index"]
    fig4 = px.bar(idx, x="day_of_week", y="index")
    fig4.add_hline(y=1.0, line_dash="dash")
    st.plotly_chart(fig4, use_container_width=True)

with col4:
    st.subheader("Trend")
    slope = linear_trend_slope(g)
    st.metric("Linear trend (units/day)", f"{slope:+.3f}")
    if slope > 0.05:
        st.caption("Demand is trending upward.")
    elif slope < -0.05:
        st.caption("Demand is trending downward.")
    else:
        st.caption("Demand is roughly flat.")

st.subheader("Demand Statistics — All SKUs")
stats = demand_statistics(cleaned)
st.dataframe(stats, use_container_width=True, hide_index=True)
