import pandas as pd
import numpy as np

from src.data.cleaner import (
    drop_duplicate_transactions,
    fill_missing_dates,
    cap_outliers,
    aggregate_daily,
    clean_pipeline,
)
from src.data.validator import validate_sales


def make_raw():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-03", "2024-01-01"]
            ),
            "sku": ["A", "A", "A", "B"],
            "quantity": [5, 3, 7, 10],
        }
    )


def test_drop_duplicate_transactions_sums_same_day():
    raw = make_raw()
    out = drop_duplicate_transactions(raw)
    a_jan1 = out[(out.sku == "A") & (out.date == "2024-01-01")]
    assert a_jan1["quantity"].iloc[0] == 8  # 5 + 3


def test_fill_missing_dates_creates_gap_free_calendar():
    raw = drop_duplicate_transactions(make_raw())
    filled = fill_missing_dates(raw)
    a = filled[filled.sku == "A"].sort_values("date")
    assert len(a) == 3  # jan1, jan2 (gap-filled), jan3
    assert a["quantity"].iloc[1] == 0.0  # jan 2 filled with zero


def test_fill_missing_dates_flags_stockout_gaps():
    raw = drop_duplicate_transactions(make_raw())
    stockouts = {"A": {pd.Timestamp("2024-01-02")}}
    filled = fill_missing_dates(raw, stockout_dates=stockouts)
    a = filled[filled.sku == "A"].sort_values("date")
    gap_row = a[a.date == pd.Timestamp("2024-01-02")]
    assert bool(gap_row["is_stockout_gap"].iloc[0]) is True


def test_cap_outliers_clips_extreme_values():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=20),
            "sku": ["A"] * 20,
            "quantity": [10.0] * 19 + [10000.0],  # one wild outlier
        }
    )
    out = cap_outliers(df, z_thresh=3.0)
    assert out["quantity"].max() < 10000.0
    assert out["was_outlier_capped"].sum() == 1


def test_aggregate_daily_is_idempotent():
    raw = drop_duplicate_transactions(make_raw())
    filled = fill_missing_dates(raw)
    agg1 = aggregate_daily(filled)
    agg2 = aggregate_daily(agg1)
    pd.testing.assert_frame_equal(
        agg1.reset_index(drop=True), agg2.reset_index(drop=True)
    )


def test_clean_pipeline_end_to_end_no_gaps_no_dupes():
    raw = make_raw()
    cleaned = clean_pipeline(raw)
    for sku, g in cleaned.groupby("sku"):
        full_range = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        assert len(g) == len(full_range)
        assert g["date"].is_monotonic_increasing
        assert not g.duplicated(subset=["date"]).any()


def test_validate_sales_flags_negative_and_missing():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4),
            "sku": ["A", "A", "A", "A"],
            "quantity": [5.0, np.nan, -2.0, 3.0],
        }
    )
    report = validate_sales(df)
    assert report.n_missing_quantity == 1
    assert report.n_negative_quantity == 1
    assert not report.is_clean()


def test_validate_sales_clean_data_has_no_warnings():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4),
            "sku": ["A", "A", "A", "A"],
            "quantity": [5.0, 2.0, 3.0, 4.0],
        }
    )
    report = validate_sales(df)
    assert report.is_clean()
    assert report.n_rows == 4
    assert report.n_skus == 1
