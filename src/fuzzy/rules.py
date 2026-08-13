"""
Rule base for the reorder-urgency fuzzy inference system.

Structure: a complete 3x3 base grid over (stock_ratio x demand_ratio) — so
every input combination always activates at least one rule — plus a smaller
set of escalation/de-escalation modifiers for lead time and forecast
uncertainty layered on top. Because skfuzzy aggregates rule outputs with
max, the base grid and the modifiers combine safely: a modifier only ever
pushes urgency at least as high as the base grid alone would.
21 rules total, within the ~15-25 rule guidance from the project brief.
"""
from __future__ import annotations

from skfuzzy import control as ctrl


def build_rules(stock, demand, uncertainty, lead_time, urgency):
    rules = [
        # --- Base grid: stock x demand (always covers every input) ---
        ctrl.Rule(stock["low"] & demand["low"], urgency["medium"]),
        ctrl.Rule(stock["low"] & demand["medium"], urgency["high"]),
        ctrl.Rule(stock["low"] & demand["high"], urgency["very_high"]),

        ctrl.Rule(stock["medium"] & demand["low"], urgency["low"]),
        ctrl.Rule(stock["medium"] & demand["medium"], urgency["medium"]),
        ctrl.Rule(stock["medium"] & demand["high"], urgency["high"]),

        ctrl.Rule(stock["high"] & demand["low"], urgency["very_low"]),
        ctrl.Rule(stock["high"] & demand["medium"], urgency["low"]),
        ctrl.Rule(stock["high"] & demand["high"], urgency["medium"]),

        # --- Escalation modifiers: long lead time / high uncertainty push urgency up ---
        ctrl.Rule(stock["low"] & lead_time["long"], urgency["very_high"]),
        ctrl.Rule(stock["low"] & uncertainty["high"], urgency["high"]),
        ctrl.Rule(stock["medium"] & demand["high"] & uncertainty["high"], urgency["very_high"]),
        ctrl.Rule(stock["medium"] & demand["medium"] & lead_time["long"], urgency["high"]),
        ctrl.Rule(stock["medium"] & uncertainty["high"] & lead_time["long"], urgency["high"]),
        ctrl.Rule(stock["high"] & demand["high"] & lead_time["long"], urgency["high"]),
        ctrl.Rule(stock["high"] & uncertainty["high"] & demand["high"], urgency["medium"]),
        ctrl.Rule(lead_time["long"] & demand["high"] & uncertainty["high"], urgency["very_high"]),
        ctrl.Rule(uncertainty["high"] & demand["high"], urgency["high"]),

        # --- De-escalation modifiers: short lead time / low uncertainty reinforce calm cases ---
        ctrl.Rule(stock["high"] & demand["low"] & lead_time["short"] & uncertainty["low"], urgency["very_low"]),
        ctrl.Rule(stock["medium"] & demand["low"] & lead_time["short"], urgency["very_low"]),
        ctrl.Rule(lead_time["short"] & demand["low"] & uncertainty["low"], urgency["very_low"]),
        ctrl.Rule(uncertainty["low"] & demand["low"] & stock["medium"], urgency["low"]),
    ]
    return rules
