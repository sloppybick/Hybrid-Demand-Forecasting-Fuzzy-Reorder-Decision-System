"""
The hybrid pipeline: Prophet forecast -> inventory features -> fuzzy engine
-> reorder urgency -> order quantity -> business recommendation.

This is the single entry point the Streamlit app (and tests) call so the
forecasting engine and fuzzy engine stay decoupled from each other, per the
architecture in the project brief.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.forecasting.prophet_model import forecast as prophet_forecast, ForecastResult
from src.inventory.reorder_point import classical_reorder_point, ReorderPointResult
from src.inventory.order_quantity import target_stock, recommended_order_quantity
from src.fuzzy.inference import ReorderFuzzyEngine, FuzzyInputs, FuzzyDecision

_EPS = 1e-6


@dataclass
class SkuRecommendation:
    sku: str
    current_stock: float
    forecast_result: ForecastResult
    lead_time_demand: float
    historical_avg_demand: float
    daily_demand_std: float
    classical: ReorderPointResult
    fuzzy: FuzzyDecision
    order_quantity_classical: float
    order_quantity_fuzzy: float

    def to_row(self) -> dict:
        return {
            "sku": self.sku,
            "current_stock": self.current_stock,
            "forecast_demand": round(self.lead_time_demand, 1),
            "lead_time": None,  # filled by caller who has inventory master
            "safety_stock": round(self.classical.safety_stock, 1),
            "reorder_point_classical": round(self.classical.reorder_point, 1),
            "decision_classical": "Reorder" if self.classical.should_reorder else "Hold",
            "reorder_urgency_fuzzy": round(self.fuzzy.urgency_score, 1),
            "decision_fuzzy": self.fuzzy.decision,
            "order_qty_classical": round(self.order_quantity_classical, 1),
            "order_qty_fuzzy": round(self.order_quantity_fuzzy, 1),
        }


def compute_recommendation(
    sku: str,
    history: pd.DataFrame,
    current_stock: float,
    lead_time_days: float,
    minimum_order_quantity: float,
    horizon: int = 14,
    service_level: float = 0.95,
    fuzzy_engine: ReorderFuzzyEngine | None = None,
) -> SkuRecommendation:
    """Run the full hybrid pipeline for a single SKU.

    history: DataFrame[date, quantity] — cleaned daily series for this SKU.
    """
    fuzzy_engine = fuzzy_engine or ReorderFuzzyEngine()

    lt_days = max(1, int(round(lead_time_days)))
    # The forecast horizon must cover at least the full lead time, or
    # lead-time demand silently truncates to whatever the horizon allows
    # (e.g. a supplier disruption that doubles lead time past a short
    # horizon would then look *less* risky, which is backwards).
    effective_horizon = max(horizon, lt_days)
    fc = prophet_forecast(sku, effective_horizon, history)
    lead_time_demand = fc.lead_time_demand(lt_days)

    historical_avg_demand = float(history["quantity"].mean())
    daily_demand_std = float(history["quantity"].std(ddof=0))

    classical = classical_reorder_point(
        current_stock=current_stock,
        lead_time_demand=lead_time_demand,
        daily_demand_std=daily_demand_std,
        lead_time_days=lead_time_days,
        service_level=service_level,
    )

    # --- Fuzzy inputs (normalized, see src/fuzzy/membership.py docstring) ---
    stock_ratio = current_stock / (lead_time_demand + _EPS)
    horizon_avg_demand = historical_avg_demand * lt_days
    demand_ratio = lead_time_demand / (horizon_avg_demand + _EPS)
    uncertainty = fc.uncertainty(index=0)

    fuzzy_inputs = FuzzyInputs(
        stock_ratio=stock_ratio,
        demand_ratio=demand_ratio,
        uncertainty=uncertainty,
        lead_time_days=lead_time_days,
    )
    fuzzy_decision = fuzzy_engine.infer(fuzzy_inputs)

    # --- Order quantities ---
    tgt_stock = target_stock(lead_time_demand, classical.safety_stock)
    order_qty_classical = recommended_order_quantity(
        current_stock, tgt_stock, minimum_order_quantity, should_order=classical.should_reorder
    )
    fuzzy_should_order = fuzzy_decision.urgency_score >= 40  # "Reorder Soon" or higher
    order_qty_fuzzy = recommended_order_quantity(
        current_stock, tgt_stock, minimum_order_quantity, should_order=fuzzy_should_order
    )

    return SkuRecommendation(
        sku=sku,
        current_stock=current_stock,
        forecast_result=fc,
        lead_time_demand=lead_time_demand,
        historical_avg_demand=historical_avg_demand,
        daily_demand_std=daily_demand_std,
        classical=classical,
        fuzzy=fuzzy_decision,
        order_quantity_classical=order_qty_classical,
        order_quantity_fuzzy=order_qty_fuzzy,
    )


def recommendations_to_dataframe(recs: list[SkuRecommendation], lead_times: dict) -> pd.DataFrame:
    rows = []
    for r in recs:
        row = r.to_row()
        row["lead_time"] = lead_times.get(r.sku)
        rows.append(row)
    return pd.DataFrame(rows)
