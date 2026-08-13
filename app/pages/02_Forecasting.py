"""Forecasting page: Prophet forecast with confidence interval + baseline comparison."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.cleaner import clean_pipeline
from src.forecasting.prophet_model import forecast as prophet_forecast
from src.forecasting.baselines import naive_forecast, moving_average_forecast, seasonal_naive_forecast

st.set_page_config(page_title="Forecasting", layout="wide")
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"


@st.cache_data(show_spinner=False)
def load():
    sales = pd.read_csv(DATA_DIR / "sales.csv")
    sales = sales.rename(columns={"quantity_sold": "quantity"})[["date", "sku", "quantity"]]
    sales["date"] = pd.to_datetime(sales["date"])
    return clean_pipeline(sales)


@st.cache_data(show_spinner=True)
def run_prophet(sku: str, horizon: int, history: pd.DataFrame):
    return prophet_forecast(sku, horizon, history)


st.title("📈 Forecasting")

cleaned = load()
col_a, col_b = st.columns([1, 2])
with col_a:
    sku = st.selectbox("SKU", sorted(cleaned["sku"].unique()))
    horizon = st.select_slider("Forecast horizon (days)", options=[7, 14, 30, 60], value=14)

history = cleaned[cleaned.sku == sku][["date", "quantity"]].sort_values("date")
result = run_prophet(sku, horizon, history)
fc_df = result.to_frame()

fig = go.Figure()
fig.add_trace(go.Scatter(x=history["date"], y=history["quantity"], name="Historical demand", line=dict(color="#7f8c8d")))
fig.add_trace(go.Scatter(x=fc_df["date"], y=fc_df["forecast_upper"], name="Upper bound", line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(
    x=fc_df["date"], y=fc_df["forecast_lower"], name="Confidence interval",
    fill="tonexty", line=dict(width=0), fillcolor="rgba(52,152,219,0.2)",
))
fig.add_trace(go.Scatter(x=fc_df["date"], y=fc_df["forecast"], name="Prophet forecast", line=dict(color="#2980b9", width=3)))
fig.update_layout(height=450, legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, use_container_width=True)

expected_demand = float(fc_df["forecast"].sum())
uncertainty = result.uncertainty(0)
unc_label = "Low" if uncertainty < 0.3 else ("Medium" if uncertainty < 0.6 else "High")

c1, c2, c3 = st.columns(3)
c1.metric(f"Expected demand ({horizon}d)", f"{expected_demand:.0f} units")
c2.metric("Day-1 forecast", f"{fc_df['forecast'].iloc[0]:.1f} units")
c3.metric("Forecast uncertainty", unc_label)

st.subheader("Baseline Comparison")
st.caption("What Prophet has to beat — see the Model Evaluation page for backtested accuracy.")
series = history["quantity"]
b1 = naive_forecast(series, horizon)
b2 = moving_average_forecast(series, horizon, k=7)
b3 = seasonal_naive_forecast(series, horizon, season_length=7)

comp_fig = go.Figure()
comp_fig.add_trace(go.Scatter(x=fc_df["date"], y=fc_df["forecast"], name="Prophet"))
comp_fig.add_trace(go.Scatter(x=fc_df["date"], y=b1, name="Naive", line=dict(dash="dot")))
comp_fig.add_trace(go.Scatter(x=fc_df["date"], y=b2, name="Moving Average", line=dict(dash="dot")))
comp_fig.add_trace(go.Scatter(x=fc_df["date"], y=b3, name="Seasonal Naive", line=dict(dash="dot")))
comp_fig.update_layout(height=350, legend=dict(orientation="h", y=1.15))
st.plotly_chart(comp_fig, use_container_width=True)
