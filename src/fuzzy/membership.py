"""
Membership functions for the fuzzy reorder-urgency engine.

Design note on normalization
-----------------------------
Raw values (units of stock, units of demand) differ by orders of magnitude
across SKUs, so the fuzzy sets are defined over *normalized, dimensionless*
inputs instead of raw units. This keeps one rule base valid for every SKU:

    stock_ratio       = current_stock / (lead_time_demand + eps)
                         1.0 = exactly enough stock to cover lead-time demand
    demand_ratio      = forecast_horizon_demand / (historical_avg_demand*horizon + eps)
                         1.0 = forecasting "business as usual"
    uncertainty_ratio = (upper - lower) / forecast     (relative forecast spread)
    lead_time_days    = raw days (kept in real units; it's already comparable
                         across SKUs since it's a supplier property, not a
                         demand-scale property)

Reorder urgency is output on a 0-100 scale.
"""
from __future__ import annotations

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Universes (ranges) for each variable
STOCK_RATIO_UNIVERSE = np.linspace(0, 3, 301)
DEMAND_RATIO_UNIVERSE = np.linspace(0, 3, 301)
UNCERTAINTY_UNIVERSE = np.linspace(0, 1.5, 151)
LEAD_TIME_UNIVERSE = np.linspace(0, 30, 301)
URGENCY_UNIVERSE = np.linspace(0, 100, 101)


def build_antecedents_consequent():
    """Construct the skfuzzy Antecedent/Consequent objects with membership sets."""
    stock = ctrl.Antecedent(STOCK_RATIO_UNIVERSE, "stock_ratio")
    demand = ctrl.Antecedent(DEMAND_RATIO_UNIVERSE, "demand_ratio")
    uncertainty = ctrl.Antecedent(UNCERTAINTY_UNIVERSE, "uncertainty")
    lead_time = ctrl.Antecedent(LEAD_TIME_UNIVERSE, "lead_time")
    urgency = ctrl.Consequent(URGENCY_UNIVERSE, "urgency")

    # Stock ratio: current_stock / lead_time_demand
    stock["low"] = fuzz.trimf(stock.universe, [0, 0, 0.8])
    stock["medium"] = fuzz.trimf(stock.universe, [0.5, 1.0, 1.6])
    stock["high"] = fuzz.trimf(stock.universe, [1.2, 3.0, 3.0])

    # Demand ratio: forecast vs historical average
    demand["low"] = fuzz.trimf(demand.universe, [0, 0, 0.8])
    demand["medium"] = fuzz.trimf(demand.universe, [0.6, 1.0, 1.4])
    demand["high"] = fuzz.trimf(demand.universe, [1.2, 3.0, 3.0])

    # Forecast uncertainty: (upper-lower)/forecast
    uncertainty["low"] = fuzz.trimf(uncertainty.universe, [0, 0, 0.3])
    uncertainty["medium"] = fuzz.trimf(uncertainty.universe, [0.2, 0.45, 0.7])
    uncertainty["high"] = fuzz.trimf(uncertainty.universe, [0.6, 1.5, 1.5])

    # Supplier lead time, days
    lead_time["short"] = fuzz.trimf(lead_time.universe, [0, 0, 7])
    lead_time["medium"] = fuzz.trimf(lead_time.universe, [5, 12, 19])
    lead_time["long"] = fuzz.trimf(lead_time.universe, [15, 30, 30])

    # Output: reorder urgency, 0-100
    urgency["very_low"] = fuzz.trimf(urgency.universe, [0, 0, 20])
    urgency["low"] = fuzz.trimf(urgency.universe, [10, 27, 45])
    urgency["medium"] = fuzz.trimf(urgency.universe, [35, 50, 65])
    urgency["high"] = fuzz.trimf(urgency.universe, [55, 73, 90])
    urgency["very_high"] = fuzz.trimf(urgency.universe, [80, 100, 100])

    return stock, demand, uncertainty, lead_time, urgency


def urgency_to_decision(score: float) -> str:
    """Map a defuzzified 0-100 urgency score to a discrete decision label."""
    if score < 20:
        return "Do Not Reorder"
    if score < 40:
        return "Monitor"
    if score < 60:
        return "Reorder Soon"
    if score < 80:
        return "Reorder Now"
    return "Urgent Reorder"
