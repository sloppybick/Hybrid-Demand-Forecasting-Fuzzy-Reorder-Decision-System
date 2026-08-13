"""
Scenario simulation: apply a demand multiplier / lead-time multiplier to a
SKU's inputs and re-run the hybrid pipeline, so the effect of a demand
spike, supplier disruption, or promotion can be compared against normal
operating conditions.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.pipeline import compute_recommendation, SkuRecommendation
from src.fuzzy.inference import ReorderFuzzyEngine

SCENARIOS = {
    "Normal": {"demand_multiplier": 1.0, "lead_time_multiplier": 1.0},
    "Demand Spike (+30%)": {"demand_multiplier": 1.3, "lead_time_multiplier": 1.0},
    "Supplier Disruption (lead time x2)": {"demand_multiplier": 1.0, "lead_time_multiplier": 2.0},
    "Promotional Period (+50%)": {"demand_multiplier": 1.5, "lead_time_multiplier": 1.0},
}


@dataclass
class ScenarioResult:
    name: str
    recommendation: SkuRecommendation


def run_scenario(
    sku: str,
    history: pd.DataFrame,
    current_stock: float,
    lead_time_days: float,
    minimum_order_quantity: float,
    scenario_name: str,
    horizon: int = 14,
    service_level: float = 0.95,
    fuzzy_engine: ReorderFuzzyEngine | None = None,
) -> ScenarioResult:
    params = SCENARIOS[scenario_name]
    scenario_history = history.copy()
    scenario_history["quantity"] = scenario_history["quantity"] * params["demand_multiplier"]
    scenario_lead_time = lead_time_days * params["lead_time_multiplier"]

    rec = compute_recommendation(
        sku=sku,
        history=scenario_history,
        current_stock=current_stock,
        lead_time_days=scenario_lead_time,
        minimum_order_quantity=minimum_order_quantity,
        horizon=horizon,
        service_level=service_level,
        fuzzy_engine=fuzzy_engine,
    )
    return ScenarioResult(name=scenario_name, recommendation=rec)


def run_all_scenarios(
    sku: str,
    history: pd.DataFrame,
    current_stock: float,
    lead_time_days: float,
    minimum_order_quantity: float,
    horizon: int = 14,
    service_level: float = 0.95,
) -> pd.DataFrame:
    engine = ReorderFuzzyEngine()
    rows = []
    for name in SCENARIOS:
        result = run_scenario(
            sku, history, current_stock, lead_time_days, minimum_order_quantity,
            name, horizon, service_level, fuzzy_engine=engine,
        )
        r = result.recommendation
        rows.append(
            {
                "scenario": name,
                "forecast_lead_time_demand": round(r.lead_time_demand, 1),
                "reorder_urgency": round(r.fuzzy.urgency_score, 1),
                "decision": r.fuzzy.decision,
                "order_qty_fuzzy": round(r.order_quantity_fuzzy, 1),
                "order_qty_classical": round(r.order_quantity_classical, 1),
            }
        )
    return pd.DataFrame(rows)
