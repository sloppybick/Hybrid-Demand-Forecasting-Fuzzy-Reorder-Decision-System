"""
Cleaning pipeline that turns raw canonical sales rows into a complete,
gap-free, per-SKU daily time series ready for feature engineering.

Pipeline order (matches the project spec):
    duplicates -> missing dates -> outliers -> daily aggregation
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def drop_duplicate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate (date, sku) rows by summing quantity."""
    return (
        df.groupby(["date", "sku"], as_index=False)["quantity"]
        .sum()
        .sort_values(["sku", "date"])
        .reset_index(drop=True)
    )


def fill_missing_dates(
    df: pd.DataFrame,
    stockout_dates: dict[str, set] | None = None,
) -> pd.DataFrame:
    """Reindex each SKU onto a complete daily calendar.

    Days with no transaction row are filled with quantity=0, BUT rows are
    tagged in `is_stockout_gap` when the caller supplies known stockout/
    closure dates for that SKU, so a modeling stage can later distinguish
    "true zero demand" from "stockout-induced zero sales" instead of
    silently treating every gap the same way.
    """
    stockout_dates = stockout_dates or {}
    filled_parts = []
    for sku, g in df.groupby("sku"):
        full_range = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(full_range)
        g["sku"] = sku
        g["quantity"] = g["quantity"].fillna(0.0)
        gap_dates = stockout_dates.get(sku, set())
        g["is_stockout_gap"] = [d in gap_dates for d in full_range]
        g.index.name = "date"
        filled_parts.append(g.reset_index())
    return pd.concat(filled_parts, ignore_index=True)


def cap_outliers(df: pd.DataFrame, z_thresh: float = 4.0) -> pd.DataFrame:
    """Cap per-SKU demand outliers at mean +/- z_thresh*std (winsorize).

    A robust rolling z-score is used per SKU so one SKU's high-volume
    baseline doesn't distort another's outlier threshold.
    """
    out = df.copy()
    capped_flags = np.zeros(len(out), dtype=bool)
    for sku, g in out.groupby("sku"):
        idx = g.index
        mu, sigma = g["quantity"].mean(), g["quantity"].std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            continue
        upper = mu + z_thresh * sigma
        lower = max(0.0, mu - z_thresh * sigma)
        mask = (g["quantity"] > upper) | (g["quantity"] < lower)
        out.loc[idx[mask], "quantity"] = out.loc[idx[mask], "quantity"].clip(
            lower=lower, upper=upper
        )
        capped_flags[idx[mask]] = True
    out["was_outlier_capped"] = capped_flags
    return out


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure exactly one row per (sku, date), summed and sorted."""
    agg_cols = {"quantity": "sum"}
    if "is_stockout_gap" in df.columns:
        agg_cols["is_stockout_gap"] = "max"
    if "was_outlier_capped" in df.columns:
        agg_cols["was_outlier_capped"] = "max"
    return (
        df.groupby(["sku", "date"], as_index=False)
        .agg(agg_cols)
        .sort_values(["sku", "date"])
        .reset_index(drop=True)
    )


def clean_pipeline(
    df: pd.DataFrame,
    stockout_dates: dict[str, set] | None = None,
    z_thresh: float = 4.0,
) -> pd.DataFrame:
    """Run the full cleaning pipeline in the documented order."""
    df = drop_duplicate_transactions(df)
    df = fill_missing_dates(df, stockout_dates=stockout_dates)
    df = cap_outliers(df, z_thresh=z_thresh)
    df = aggregate_daily(df)
    return df
