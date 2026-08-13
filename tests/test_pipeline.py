import pandas as pd
import numpy as np
import pytest

from src.pipeline import compute_recommendation
from src.simulation.scenarios import run_all_scenarios, SCENARIOS
from src.fuzzy.inference import ReorderFuzzyEngine


def make_history(n_days=120, base=50, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    dow = dates.dayofweek.values
    weekly = 1 + 0.2 * np.where(dow >= 5, 1.0, -0.2)
    quantity = np.clip(base * weekly + rng.normal(0, 5, n_days), 0, None)
    return pd.DataFrame({"date": dates, "quantity": quantity})


@pytest.fixture(scope="module")
def history_low_stock_case():
    return make_history(base=60, seed=2)


def test_compute_recommendation_returns_consistent_shapes(history_low_stock_case):
    rec = compute_recommendation(
        sku="TEST-1",
        history=history_low_stock_case,
        current_stock=30,
        lead_time_days=7,
        minimum_order_quantity=20,
        horizon=14,
    )
    assert rec.sku == "TEST-1"
    assert len(rec.forecast_result.predicted_demand) == 14
    assert rec.lead_time_demand > 0
    assert 0 <= rec.fuzzy.urgency_score <= 100
    assert rec.order_quantity_classical >= 0
    assert rec.order_quantity_fuzzy >= 0


def test_low_stock_triggers_reorder(history_low_stock_case):
    rec = compute_recommendation(
        sku="TEST-LOW",
        history=history_low_stock_case,
        current_stock=5,  # very low vs demand of ~60/day
        lead_time_days=10,
        minimum_order_quantity=10,
        horizon=14,
    )
    assert rec.classical.should_reorder is True
    assert rec.fuzzy.decision in {"Reorder Now", "Urgent Reorder", "Reorder Soon"}
    assert rec.order_quantity_fuzzy > 0


def test_high_stock_does_not_trigger_reorder(history_low_stock_case):
    rec = compute_recommendation(
        sku="TEST-HIGH",
        history=history_low_stock_case,
        current_stock=5000,
        lead_time_days=5,
        minimum_order_quantity=10,
        horizon=14,
    )
    assert rec.classical.should_reorder is False
    assert rec.order_quantity_classical == 0
    assert rec.fuzzy.urgency_score < 50


def test_row_dict_has_expected_keys(history_low_stock_case):
    rec = compute_recommendation(
        sku="TEST-ROW",
        history=history_low_stock_case,
        current_stock=100,
        lead_time_days=7,
        minimum_order_quantity=20,
        horizon=14,
    )
    row = rec.to_row()
    expected = {
        "sku", "current_stock", "forecast_demand", "lead_time", "safety_stock",
        "reorder_point_classical", "decision_classical", "reorder_urgency_fuzzy",
        "decision_fuzzy", "order_qty_classical", "order_qty_fuzzy",
    }
    assert expected.issubset(row.keys())


def test_run_all_scenarios_covers_every_scenario(history_low_stock_case):
    df = run_all_scenarios(
        sku="TEST-SCN",
        history=history_low_stock_case,
        current_stock=80,
        lead_time_days=7,
        minimum_order_quantity=20,
        horizon=10,
    )
    assert set(df["scenario"]) == set(SCENARIOS.keys())
    assert len(df) == len(SCENARIOS)


def test_supplier_disruption_scenario_increases_or_maintains_urgency(history_low_stock_case):
    df = run_all_scenarios(
        sku="TEST-SCN2",
        history=history_low_stock_case,
        current_stock=80,
        lead_time_days=7,
        minimum_order_quantity=20,
        horizon=10,
    )
    normal = df[df.scenario == "Normal"]["reorder_urgency"].iloc[0]
    disrupted = df[df.scenario == "Supplier Disruption (lead time x2)"]["reorder_urgency"].iloc[0]
    assert disrupted >= normal - 1e-6


def test_reused_fuzzy_engine_gives_same_result_as_fresh_one(history_low_stock_case):
    engine = ReorderFuzzyEngine()
    rec_a = compute_recommendation(
        sku="A", history=history_low_stock_case, current_stock=40,
        lead_time_days=7, minimum_order_quantity=10, horizon=10, fuzzy_engine=engine,
    )
    rec_b = compute_recommendation(
        sku="A", history=history_low_stock_case, current_stock=40,
        lead_time_days=7, minimum_order_quantity=10, horizon=10,
    )
    assert rec_a.fuzzy.urgency_score == pytest.approx(rec_b.fuzzy.urgency_score, abs=0.5)
