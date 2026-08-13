import numpy as np
import pandas as pd
import pytest

from src.forecasting.baselines import (
    naive_forecast,
    moving_average_forecast,
    seasonal_naive_forecast,
)
from src.forecasting.evaluation import (
    mae, rmse, mape, smape, all_metrics,
    expanding_window_backtest, summarize_folds,
)


@pytest.fixture
def flat_series():
    return pd.Series([10.0] * 30)


@pytest.fixture
def weekly_series():
    # repeating weekly pattern of length 7
    pattern = [10, 12, 14, 11, 13, 20, 22]
    return pd.Series(pattern * 6)  # 42 days


def test_naive_forecast_repeats_last_value(flat_series):
    out = naive_forecast(flat_series, horizon=5)
    assert len(out) == 5
    assert np.allclose(out, 10.0)


def test_naive_forecast_uses_true_last_value():
    s = pd.Series([1, 2, 3, 100])
    out = naive_forecast(s, horizon=3)
    assert np.allclose(out, 100.0)


def test_moving_average_forecast(flat_series):
    out = moving_average_forecast(flat_series, horizon=4, k=5)
    assert len(out) == 4
    assert np.allclose(out, 10.0)


def test_moving_average_handles_short_history():
    s = pd.Series([2, 4])
    out = moving_average_forecast(s, horizon=2, k=10)
    assert np.allclose(out, 3.0)  # k gets clamped to len(history)


def test_seasonal_naive_repeats_last_cycle(weekly_series):
    out = seasonal_naive_forecast(weekly_series, horizon=7, season_length=7)
    assert len(out) == 7
    assert np.allclose(out, [10, 12, 14, 11, 13, 20, 22])


def test_seasonal_naive_tiles_for_longer_horizon(weekly_series):
    out = seasonal_naive_forecast(weekly_series, horizon=10, season_length=7)
    assert len(out) == 10
    assert np.allclose(out[:3], [10, 12, 14])


def test_seasonal_naive_falls_back_to_naive_when_short_history():
    s = pd.Series([5, 6, 7])
    out = seasonal_naive_forecast(s, horizon=3, season_length=7)
    assert np.allclose(out, 7.0)


def test_metrics_zero_error():
    y = np.array([10.0, 20.0, 30.0])
    m = all_metrics(y, y)
    assert m["MAE"] == 0
    assert m["RMSE"] == 0
    assert m["MAPE"] == 0
    assert m["sMAPE"] == 0


def test_mae_basic():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 33])
    assert mae(y_true, y_pred) == pytest.approx((2 + 2 + 3) / 3)


def test_rmse_ge_mae():
    y_true = np.array([10, 20, 30, 5])
    y_pred = np.array([12, 18, 40, 5])
    assert rmse(y_true, y_pred) >= mae(y_true, y_pred)


def test_smape_bounded_0_200():
    y_true = np.array([0.0, 5.0, 100.0])
    y_pred = np.array([10.0, 0.0, 1.0])
    val = smape(y_true, y_pred)
    assert 0 <= val <= 200


def test_backtest_naive_produces_folds():
    series = pd.Series(np.tile([10, 12, 14, 11, 13, 20, 22], 10))  # 70 days
    dates = pd.Series(pd.date_range("2024-01-01", periods=70))
    folds = expanding_window_backtest(
        series, dates, naive_forecast, horizon=7, n_folds=3, min_train_size=35
    )
    assert len(folds) == 3
    for f in folds:
        assert set(f.metrics.keys()) == {"MAE", "RMSE", "MAPE", "sMAPE"}
    summary = summarize_folds(folds)
    assert set(summary.keys()) == {"MAE", "RMSE", "MAPE", "sMAPE"}


def test_backtest_stops_when_not_enough_data():
    series = pd.Series(np.arange(20, dtype=float))
    dates = pd.Series(pd.date_range("2024-01-01", periods=20))
    folds = expanding_window_backtest(
        series, dates, naive_forecast, horizon=7, n_folds=5, min_train_size=15
    )
    # only room for a couple folds before running out of data
    assert len(folds) <= 2
