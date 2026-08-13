"""Order-quantity sizing shared by both the classical and fuzzy policies."""
from __future__ import annotations


def target_stock(forecast_demand_horizon: float, safety_stock: float) -> float:
    """TargetStock = ForecastDemand(horizon) + SafetyStock"""
    return forecast_demand_horizon + safety_stock


def recommended_order_quantity(
    current_stock: float,
    target_stock_level: float,
    minimum_order_quantity: float = 0.0,
    should_order: bool = True,
) -> float:
    """OrderQty = max(0, TargetStock - CurrentStock), floored at MOQ once a
    reorder is actually recommended (never forces an order that isn't needed).
    """
    if not should_order:
        return 0.0
    raw_qty = max(0.0, target_stock_level - current_stock)
    if raw_qty <= 0:
        return 0.0
    return max(raw_qty, minimum_order_quantity)
