"""Simple forecasting baselines that Prophet has to beat to justify its cost."""
from __future__ import annotations

import numpy as np
import pandas as pd


def naive_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    """Repeat the last observed value for every step of the horizon."""
    last = history.iloc[-1]
    return np.full(horizon, last, dtype=float)


def moving_average_forecast(history: pd.Series, horizon: int, k: int = 7) -> np.ndarray:
    """Repeat the mean of the last k observations for every step of the horizon."""
    k = min(k, len(history))
    avg = history.iloc[-k:].mean()
    return np.full(horizon, avg, dtype=float)


def seasonal_naive_forecast(history: pd.Series, horizon: int, season_length: int = 7) -> np.ndarray:
    """Repeat the last full seasonal cycle (default weekly, s=7) forward."""
    s = season_length
    if len(history) < s:
        return naive_forecast(history, horizon)
    last_season = history.iloc[-s:].values
    reps = int(np.ceil(horizon / s))
    tiled = np.tile(last_season, reps)[:horizon]
    return tiled


BASELINES = {
    "naive": naive_forecast,
    "moving_average": moving_average_forecast,
    "seasonal_naive": seasonal_naive_forecast,
}
