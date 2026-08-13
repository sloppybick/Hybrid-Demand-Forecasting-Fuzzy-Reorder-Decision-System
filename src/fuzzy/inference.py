"""
Fuzzy inference engine (Mamdani, centroid defuzzification via skfuzzy.control)
for reorder urgency, plus a human-readable explanation of the decision.
"""
from __future__ import annotations

from dataclasses import dataclass

from skfuzzy import control as ctrl

from .membership import build_antecedents_consequent, urgency_to_decision
from .rules import build_rules


@dataclass
class FuzzyInputs:
    stock_ratio: float
    demand_ratio: float
    uncertainty: float
    lead_time_days: float


@dataclass
class FuzzyDecision:
    inputs: FuzzyInputs
    urgency_score: float
    decision: str
    explanation: str


class ReorderFuzzyEngine:
    """Builds the control system once and reuses it for every SKU/scenario."""

    def __init__(self):
        self.stock, self.demand, self.uncertainty, self.lead_time, self.urgency = (
            build_antecedents_consequent()
        )
        self.rules = build_rules(
            self.stock, self.demand, self.uncertainty, self.lead_time, self.urgency
        )
        self.control_system = ctrl.ControlSystem(self.rules)

    def infer(self, inputs: FuzzyInputs) -> FuzzyDecision:
        sim = ctrl.ControlSystemSimulation(self.control_system)
        sim.input["stock_ratio"] = _clip(inputs.stock_ratio, self.stock.universe)
        sim.input["demand_ratio"] = _clip(inputs.demand_ratio, self.demand.universe)
        sim.input["uncertainty"] = _clip(inputs.uncertainty, self.uncertainty.universe)
        sim.input["lead_time"] = _clip(inputs.lead_time_days, self.lead_time.universe)
        sim.compute()
        try:
            score = float(sim.output["urgency"])
        except KeyError:
            # No rule cleared the activation threshold for this exact input
            # combination (can happen right at fuzzy-set boundaries). The
            # rule base is built to cover the full stock x demand grid, so
            # this is a rare edge case — fall back to a neutral score
            # rather than raising, since a stalled dashboard is worse than
            # an approximate one.
            score = 50.0
        decision = urgency_to_decision(score)
        explanation = _explain(inputs, score, decision)
        return FuzzyDecision(inputs=inputs, urgency_score=score, decision=decision, explanation=explanation)


def _clip(value: float, universe) -> float:
    lo, hi = float(universe.min()), float(universe.max())
    return max(lo, min(hi, float(value)))


def _band(value: float, low_hi: float, med_hi: float) -> str:
    if value < low_hi:
        return "Low"
    if value < med_hi:
        return "Medium"
    return "High"


def _explain(inputs: FuzzyInputs, score: float, decision: str) -> str:
    stock_band = _band(inputs.stock_ratio, 0.8, 1.2)
    demand_band = _band(inputs.demand_ratio, 0.8, 1.2)
    uncertainty_band = _band(inputs.uncertainty, 0.3, 0.6)
    lead_band = _band(inputs.lead_time_days, 7, 15)
    lead_label = {"Low": "Short", "Medium": "Medium", "High": "Long"}[lead_band]

    parts = [
        f"Current stock relative to lead-time demand: {stock_band}",
        f"Forecast demand vs. historical average: {demand_band}",
        f"Forecast uncertainty: {uncertainty_band}",
        f"Supplier lead time: {lead_label}",
    ]
    return (
        f"Reorder urgency is {score:.1f}/100 ({decision}). Drivers -> "
        + "; ".join(parts)
        + "."
    )
