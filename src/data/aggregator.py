"""
Exploratory demand statistics per SKU: dispersion, intermittency, and simple
seasonality/trend signals used both for the EDA page and as sanity context
next to the Prophet forecast ("why did the model choose this?").
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def demand_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """One row of summary statistics per SKU.

    df must be a cleaned, gap-free daily series with columns [sku, date, quantity].
    """
    rows = []
    for sku, g in df.groupby("sku"):
        q = g["quantity"].values
        mean = float(np.mean(q))
        std = float(np.std(q, ddof=0))
        cov = std / mean if mean > 0 else np.nan
        intermittency = float(np.mean(q == 0))
        rows.append(
            {
                "sku": sku,
                "mean_demand": mean,
                "median_demand": float(np.median(q)),
                "std_demand": std,
                "coefficient_of_variation": cov,
                "min_demand": float(np.min(q)),
                "max_demand": float(np.max(q)),
                "intermittency": intermittency,
                "n_days": len(q),
            }
        )
    return pd.DataFrame(rows).sort_values("sku").reset_index(drop=True)


def rolling_features(g: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Rolling mean/std for a single SKU's daily series (sorted by date)."""
    g = g.sort_values("date").copy()
    g[f"rolling_mean_{window}d"] = g["quantity"].rolling(window, min_periods=1).mean()
    g[f"rolling_std_{window}d"] = g["quantity"].rolling(window, min_periods=1).std().fillna(0.0)
    return g


def weekly_seasonality_index(g: pd.DataFrame) -> pd.Series:
    """Average demand by day-of-week, normalized to overall mean (1.0 = average)."""
    g = g.copy()
    g["dow"] = pd.to_datetime(g["date"]).dt.day_name()
    overall_mean = g["quantity"].mean()
    if overall_mean == 0:
        return g.groupby("dow")["quantity"].mean() * 0.0
    idx = g.groupby("dow")["quantity"].mean() / overall_mean
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return idx.reindex(order)


def linear_trend_slope(g: pd.DataFrame) -> float:
    """Slope of a simple linear fit of quantity over time (units/day)."""
    g = g.sort_values("date")
    t = np.arange(len(g))
    if len(g) < 2:
        return 0.0
    slope, _ = np.polyfit(t, g["quantity"].values, 1)
    return float(slope)
