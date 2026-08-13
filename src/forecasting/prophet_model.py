"""Prophet-based per-SKU demand forecasting engine.

Wraps Prophet behind a small, stable interface (`forecast`) so the rest of
the system (inventory math, fuzzy engine, dashboard) never touches Prophet
directly and the forecasting backend could be swapped later.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


@dataclass
class ForecastResult:
    dates: pd.DatetimeIndex
    predicted_demand: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": self.dates,
                "forecast": self.predicted_demand,
                "forecast_lower": self.lower_bound,
                "forecast_upper": self.upper_bound,
            }
        )

    def lead_time_demand(self, lead_time_days: int) -> float:
        """Sum of forecast demand over the first `lead_time_days` of the horizon."""
        n = min(lead_time_days, len(self.predicted_demand))
        return float(np.sum(self.predicted_demand[:n]))

    def uncertainty(self, index: int = 0) -> float:
        """Normalized forecast uncertainty at a given horizon step: (upper-lower)/forecast."""
        f = self.predicted_demand[index]
        if f <= 0:
            return 0.0
        return float((self.upper_bound[index] - self.lower_bound[index]) / f)


def _build_model(weekly_seasonality: bool = True, yearly_seasonality: bool = "auto",
                  interval_width: float = 0.8):
    from prophet import Prophet

    return Prophet(
        daily_seasonality=False,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        interval_width=interval_width,
    )


def fit_prophet(history: pd.DataFrame):
    """history: DataFrame with columns [date, quantity] for a single SKU."""
    df = pd.DataFrame({"ds": pd.to_datetime(history["date"]), "y": history["quantity"].astype(float)})
    model = _build_model()
    model.fit(df)
    return model


def forecast(sku: str, horizon: int, history: pd.DataFrame) -> ForecastResult:
    """Fit Prophet on `history` and forecast `horizon` days ahead for one SKU.

    history: DataFrame with columns [date, quantity], already daily & gap-free.
    """
    model = fit_prophet(history)
    future = model.make_future_dataframe(periods=horizon, freq="D")
    fcst = model.predict(future)
    tail = fcst.tail(horizon)
    yhat = np.clip(tail["yhat"].values, a_min=0, a_max=None)
    lower = np.clip(tail["yhat_lower"].values, a_min=0, a_max=None)
    upper = np.clip(tail["yhat_upper"].values, a_min=0, a_max=None)
    return ForecastResult(
        dates=pd.DatetimeIndex(tail["ds"]),
        predicted_demand=yhat,
        lower_bound=lower,
        upper_bound=upper,
    )


def forecast_from_series(history: pd.Series, horizon: int, freq_start: pd.Timestamp | None = None) -> np.ndarray:
    """Adapter matching the baseline signature (history: pd.Series) for backtesting.

    Builds a synthetic date index since backtesting only cares about the
    values, not real calendar dates.
    """
    dates = pd.date_range("2000-01-01", periods=len(history), freq="D")
    hist_df = pd.DataFrame({"date": dates, "quantity": history.values})
    result = forecast("BACKTEST", horizon, hist_df)
    return result.predicted_demand
