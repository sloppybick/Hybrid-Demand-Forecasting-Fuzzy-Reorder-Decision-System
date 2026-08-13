"""Classical safety-stock and lead-time demand calculations."""
from __future__ import annotations

import numpy as np

# Standard normal z-values for common service levels, used as the
# safety-stock multiplier z in: SS = z * sigma_LT
SERVICE_LEVEL_Z = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.975: 1.9600,
    0.98: 2.0537,
    0.99: 2.3263,
    0.995: 2.5758,
}


def z_for_service_level(service_level: float) -> float:
    """Nearest-match z-score for an arbitrary service level (0-1)."""
    if service_level in SERVICE_LEVEL_Z:
        return SERVICE_LEVEL_Z[service_level]
    # simple monotonic interpolation over the known table
    levels = sorted(SERVICE_LEVEL_Z)
    if service_level <= levels[0]:
        return SERVICE_LEVEL_Z[levels[0]]
    if service_level >= levels[-1]:
        return SERVICE_LEVEL_Z[levels[-1]]
    for lo, hi in zip(levels, levels[1:]):
        if lo <= service_level <= hi:
            z_lo, z_hi = SERVICE_LEVEL_Z[lo], SERVICE_LEVEL_Z[hi]
            frac = (service_level - lo) / (hi - lo)
            return z_lo + frac * (z_hi - z_lo)
    return 1.6449  # fallback: 95%


def lead_time_demand_std(daily_std: float, lead_time_days: float) -> float:
    """Std dev of demand over the lead time, assuming i.i.d. daily demand:
    sigma_LT = sigma_daily * sqrt(L)
    """
    return float(daily_std * np.sqrt(max(lead_time_days, 0)))


def safety_stock(daily_std: float, lead_time_days: float, service_level: float = 0.95) -> float:
    """SS = z * sigma_daily * sqrt(L)"""
    z = z_for_service_level(service_level)
    return float(z * lead_time_demand_std(daily_std, lead_time_days))
