"""Model Evaluation: rolling-origin backtest comparing baselines against Prophet."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.cleaner import clean_pipeline
from src.forecasting.baselines import BASELINES
from src.forecasting.prophet_model import forecast_from_series
from src.forecasting.evaluation import expanding_window_backtest, summarize_folds

st.set_page_config(page_title="Model Evaluation", layout="wide")
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"


@st.cache_data(show_spinner=False)
def load():
    sales = pd.read_csv(DATA_DIR / "sales.csv")
    sales = sales.rename(columns={"quantity_sold": "quantity"})[["date", "sku", "quantity"]]
    sales["date"] = pd.to_datetime(sales["date"])
    return clean_pipeline(sales)


st.title("📊 Model Evaluation")
st.caption("Rolling-origin (expanding window) backtest: MAE / RMSE / MAPE / sMAPE, averaged across folds.")

cleaned = load()
sku = st.selectbox("SKU", sorted(cleaned["sku"].unique()))
horizon = st.select_slider("Test horizon per fold (days)", options=[7, 14], value=7)
n_folds = st.slider("Number of folds", 2, 5, 3)

g = cleaned[cleaned.sku == sku].sort_values("date")
series = g["quantity"].reset_index(drop=True)
dates = g["date"].reset_index(drop=True)

run = st.button("Run backtest", type="primary")

if run:
    results = {}
    with st.spinner("Backtesting baselines..."):
        for name, fn in BASELINES.items():
            folds = expanding_window_backtest(series, dates, fn, horizon=horizon, n_folds=n_folds)
            results[name.replace("_", " ").title()] = summarize_folds(folds)

    with st.spinner("Backtesting Prophet (this fits a model per fold, slower)..."):
        folds = expanding_window_backtest(series, dates, forecast_from_series, horizon=horizon, n_folds=n_folds)
        results["Prophet"] = summarize_folds(folds)

    summary_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    st.subheader("Backtested Accuracy")
    st.dataframe(summary_df.round(2), use_container_width=True, hide_index=True)

    best_model = summary_df.loc[summary_df["MAE"].idxmin(), "model"]
    seasonal_mae = summary_df.loc[summary_df.model == "Seasonal Naive", "MAE"]
    prophet_mae = summary_df.loc[summary_df.model == "Prophet", "MAE"]
    if not seasonal_mae.empty and not prophet_mae.empty and seasonal_mae.iloc[0] > 0:
        improvement = (1 - prophet_mae.iloc[0] / seasonal_mae.iloc[0]) * 100
        if improvement > 0:
            st.success(f"Prophet reduced MAE by {improvement:.1f}% vs. the seasonal-naive baseline.")
        else:
            st.warning(f"Prophet's MAE was {-improvement:.1f}% worse than seasonal-naive on this SKU/window.")
    st.info(f"Lowest MAE this run: **{best_model}**")

    fig = px.bar(summary_df, x="model", y="MAE", color="model")
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Click **Run backtest** to compare Naive, Moving Average, Seasonal Naive, and Prophet.")
