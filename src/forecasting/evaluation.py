"""Forecast accuracy metrics and rolling-origin (expanding window) backtesting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < eps, eps, denom)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100)


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }


@dataclass
class BacktestFold:
    fold: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    metrics: dict


def expanding_window_backtest(
    series: pd.Series,
    dates: pd.Series,
    forecast_fn,
    horizon: int = 14,
    n_folds: int = 4,
    min_train_size: int | None = None,
) -> list[BacktestFold]:
    """Rolling-origin backtest: train grows each fold, test window is fixed.

    forecast_fn(history: pd.Series, horizon: int) -> np.ndarray
    Works for baselines directly; Prophet is wrapped to match this signature
    (see prophet_model.forecast_from_series).
    """
    n = len(series)
    min_train_size = min_train_size or max(30, n - n_folds * horizon)
    folds = []
    fold_starts = [min_train_size + i * horizon for i in range(n_folds)]

    for i, train_end_idx in enumerate(fold_starts):
        test_end_idx = train_end_idx + horizon
        if test_end_idx > n:
            break
        train = series.iloc[:train_end_idx]
        y_true = series.iloc[train_end_idx:test_end_idx].values
        y_pred = forecast_fn(train, horizon)
        m = all_metrics(y_true, y_pred)
        folds.append(
            BacktestFold(
                fold=i,
                train_end=dates.iloc[train_end_idx - 1],
                test_start=dates.iloc[train_end_idx],
                test_end=dates.iloc[test_end_idx - 1],
                metrics=m,
            )
        )
    return folds


def summarize_folds(folds: list[BacktestFold]) -> dict:
    """Average each metric across folds."""
    if not folds:
        return {}
    keys = folds[0].metrics.keys()
    return {k: float(np.mean([f.metrics[k] for f in folds])) for k in keys}
