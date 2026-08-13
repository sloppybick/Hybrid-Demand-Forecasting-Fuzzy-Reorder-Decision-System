"""Classical reorder-point (ROP) policy — the baseline the fuzzy system competes against."""
from __future__ import annotations

from dataclasses import dataclass

from .safety_stock import safety_stock


@dataclass
class ReorderPointResult:
    lead_time_demand: float
    safety_stock: float
    reorder_point: float
    should_reorder: bool


def classical_reorder_point(
    current_stock: float,
    lead_time_demand: float,
    daily_demand_std: float,
    lead_time_days: float,
    service_level: float = 0.95,
) -> ReorderPointResult:
    """ROP = expected lead-time demand + safety stock.

    Reorder triggers when current_stock <= ROP (the textbook (s, Q) policy).
    """
    ss = safety_stock(daily_demand_std, lead_time_days, service_level)
    rop = lead_time_demand + ss
    return ReorderPointResult(
        lead_time_demand=lead_time_demand,
        safety_stock=ss,
        reorder_point=rop,
        should_reorder=current_stock <= rop,
    )
