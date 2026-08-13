import pytest

from src.inventory.safety_stock import (
    z_for_service_level,
    lead_time_demand_std,
    safety_stock,
    SERVICE_LEVEL_Z,
)
from src.inventory.reorder_point import classical_reorder_point
from src.inventory.order_quantity import target_stock, recommended_order_quantity


def test_z_for_known_service_level():
    assert z_for_service_level(0.95) == pytest.approx(1.6449)


def test_z_interpolates_between_known_levels():
    z = z_for_service_level(0.9625)  # halfway between 0.95 and 0.975
    lo, hi = SERVICE_LEVEL_Z[0.95], SERVICE_LEVEL_Z[0.975]
    assert lo < z < hi


def test_z_clamps_outside_table():
    assert z_for_service_level(0.5) == SERVICE_LEVEL_Z[min(SERVICE_LEVEL_Z)]
    assert z_for_service_level(0.9999) == SERVICE_LEVEL_Z[max(SERVICE_LEVEL_Z)]


def test_lead_time_demand_std_scales_with_sqrt_lead_time():
    std_7 = lead_time_demand_std(daily_std=5.0, lead_time_days=7)
    std_28 = lead_time_demand_std(daily_std=5.0, lead_time_days=28)
    # doubling sqrt(L) by quadrupling L
    assert std_28 == pytest.approx(std_7 * 2, rel=1e-6)


def test_safety_stock_higher_service_level_means_more_stock():
    ss_low = safety_stock(daily_std=4.0, lead_time_days=7, service_level=0.90)
    ss_high = safety_stock(daily_std=4.0, lead_time_days=7, service_level=0.99)
    assert ss_high > ss_low


def test_classical_reorder_point_triggers_when_low_stock():
    result = classical_reorder_point(
        current_stock=20,
        lead_time_demand=100,
        daily_demand_std=10,
        lead_time_days=7,
        service_level=0.95,
    )
    assert result.should_reorder is True
    assert result.reorder_point > result.lead_time_demand  # safety stock adds on top


def test_classical_reorder_point_holds_when_well_stocked():
    result = classical_reorder_point(
        current_stock=500,
        lead_time_demand=100,
        daily_demand_std=10,
        lead_time_days=7,
        service_level=0.95,
    )
    assert result.should_reorder is False


def test_target_stock_combines_forecast_and_safety():
    assert target_stock(forecast_demand_horizon=150, safety_stock=30) == 180


def test_recommended_order_quantity_zero_when_not_ordering():
    qty = recommended_order_quantity(
        current_stock=100, target_stock_level=150, minimum_order_quantity=20, should_order=False
    )
    assert qty == 0.0


def test_recommended_order_quantity_respects_moq():
    qty = recommended_order_quantity(
        current_stock=140, target_stock_level=150, minimum_order_quantity=50, should_order=True
    )
    # raw need is only 10 units, but MOQ forces 50
    assert qty == 50.0


def test_recommended_order_quantity_no_moq_needed_when_gap_large():
    qty = recommended_order_quantity(
        current_stock=10, target_stock_level=150, minimum_order_quantity=20, should_order=True
    )
    assert qty == 140.0


def test_recommended_order_quantity_zero_when_already_overstocked():
    qty = recommended_order_quantity(
        current_stock=300, target_stock_level=150, minimum_order_quantity=20, should_order=True
    )
    assert qty == 0.0
