"""
Generates a synthetic but realistic multi-SKU retail dataset (trend +
weekly seasonality + noise + occasional promotions) so the system is fully
runnable end-to-end without requiring an external Kaggle download first.
Swap in a real dataset later via src/data/loader.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SKU_PROFILES = {
    "SKU-001": {"base": 40, "trend": 0.02, "weekly_amp": 0.25, "noise": 0.15, "lead_time": 7, "moq": 50, "unit_cost": 4.2},
    "SKU-002": {"base": 15, "trend": -0.01, "weekly_amp": 0.10, "noise": 0.30, "lead_time": 5, "moq": 20, "unit_cost": 12.5},
    "SKU-003": {"base": 80, "trend": 0.05, "weekly_amp": 0.35, "noise": 0.10, "lead_time": 10, "moq": 100, "unit_cost": 2.1},
    "SKU-004": {"base": 8, "trend": 0.00, "weekly_amp": 0.05, "noise": 0.50, "lead_time": 14, "moq": 10, "unit_cost": 35.0},
    "SKU-005": {"base": 55, "trend": 0.01, "weekly_amp": 0.20, "noise": 0.20, "lead_time": 6, "moq": 60, "unit_cost": 6.8},
}


def generate_sales(
    start_date: str = "2024-01-01",
    n_days: int = 540,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    rows = []

    for sku, prof in SKU_PROFILES.items():
        t = np.arange(n_days)
        trend = prof["base"] * (1 + prof["trend"] * t / 30.0)
        dow = dates.dayofweek.values  # 0=Mon
        weekly = 1 + prof["weekly_amp"] * np.where(dow >= 5, 1.0, -0.3) * np.sin(np.pi * (dow + 1) / 7)
        promo_days = rng.random(n_days) < 0.04
        promo_boost = np.where(promo_days, rng.uniform(1.3, 1.8, n_days), 1.0)
        mean_demand = np.clip(trend * weekly * promo_boost, 0.5, None)
        noise = rng.normal(0, prof["noise"], n_days)
        quantity = np.clip(mean_demand * (1 + noise), 0, None)
        quantity = rng.poisson(np.clip(quantity, 0.01, None)).astype(float)

        for d, q, promo in zip(dates, quantity, promo_days):
            rows.append(
                {
                    "date": d,
                    "sku": sku,
                    "quantity_sold": q,
                    "promotion": bool(promo),
                }
            )

    return pd.DataFrame(rows)


def generate_inventory_master() -> pd.DataFrame:
    rows = []
    for sku, prof in SKU_PROFILES.items():
        rows.append(
            {
                "sku": sku,
                "current_stock": None,  # filled per-scenario by the app / simulator
                "lead_time": prof["lead_time"],
                "minimum_order_quantity": prof["moq"],
                "unit_cost": prof["unit_cost"],
            }
        )
    return pd.DataFrame(rows)
