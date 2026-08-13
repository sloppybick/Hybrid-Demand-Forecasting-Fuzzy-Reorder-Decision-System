import pytest

from src.fuzzy.inference import ReorderFuzzyEngine, FuzzyInputs
from src.fuzzy.membership import urgency_to_decision


@pytest.fixture(scope="module")
def engine():
    return ReorderFuzzyEngine()


def test_urgency_to_decision_bands():
    assert urgency_to_decision(5) == "Do Not Reorder"
    assert urgency_to_decision(25) == "Monitor"
    assert urgency_to_decision(45) == "Reorder Soon"
    assert urgency_to_decision(65) == "Reorder Now"
    assert urgency_to_decision(95) == "Urgent Reorder"


def test_low_stock_high_demand_gives_high_urgency(engine):
    result = engine.infer(
        FuzzyInputs(stock_ratio=0.2, demand_ratio=1.6, uncertainty=0.2, lead_time_days=7)
    )
    assert result.urgency_score >= 60
    assert result.decision in {"Reorder Now", "Urgent Reorder"}


def test_high_stock_low_demand_gives_low_urgency(engine):
    result = engine.infer(
        FuzzyInputs(stock_ratio=2.5, demand_ratio=0.3, uncertainty=0.1, lead_time_days=3)
    )
    assert result.urgency_score <= 40
    assert result.decision in {"Do Not Reorder", "Monitor"}


def test_low_stock_long_lead_time_is_urgent_even_with_low_demand(engine):
    result = engine.infer(
        FuzzyInputs(stock_ratio=0.3, demand_ratio=0.5, uncertainty=0.2, lead_time_days=28)
    )
    assert result.urgency_score >= 50


def test_increasing_uncertainty_does_not_decrease_urgency(engine):
    base = engine.infer(
        FuzzyInputs(stock_ratio=1.0, demand_ratio=1.3, uncertainty=0.1, lead_time_days=10)
    )
    more_uncertain = engine.infer(
        FuzzyInputs(stock_ratio=1.0, demand_ratio=1.3, uncertainty=1.0, lead_time_days=10)
    )
    assert more_uncertain.urgency_score >= base.urgency_score - 1e-6


def test_inputs_are_clipped_to_universe_without_error(engine):
    # values far outside the defined universes should not raise
    result = engine.infer(
        FuzzyInputs(stock_ratio=-5, demand_ratio=999, uncertainty=-1, lead_time_days=1000)
    )
    assert 0 <= result.urgency_score <= 100


def test_explanation_mentions_all_four_drivers(engine):
    result = engine.infer(
        FuzzyInputs(stock_ratio=0.5, demand_ratio=1.1, uncertainty=0.4, lead_time_days=12)
    )
    text = result.explanation.lower()
    assert "stock" in text
    assert "demand" in text
    assert "uncertainty" in text
    assert "lead time" in text


def test_urgency_score_within_bounds_across_grid(engine):
    import itertools

    for stock, demand, unc, lt in itertools.product(
        [0.1, 1.0, 2.5], [0.2, 1.0, 2.0], [0.05, 0.5, 1.2], [2, 10, 25]
    ):
        result = engine.infer(
            FuzzyInputs(stock_ratio=stock, demand_ratio=demand, uncertainty=unc, lead_time_days=lt)
        )
        assert 0 <= result.urgency_score <= 100
