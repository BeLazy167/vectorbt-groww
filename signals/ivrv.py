"""IV/RV ratio computation."""

from __future__ import annotations

import pandas as pd

from data.iv_engine import realized_vol


def compute_ivrv(atm_iv: float, underlying_prices: pd.Series, rv_window: int = 30) -> float:
    """Compute IV/RV ratio. Ratio > 1 means IV overstates realized vol."""
    rv_series = realized_vol(underlying_prices, rv_window).dropna()
    if rv_series.empty:
        return 0.0
    rv = rv_series.iloc[-1]
    if rv == 0:
        return 0.0
    return atm_iv / rv


def compute_ivrv_series(
    iv_series: pd.Series, underlying_prices: pd.Series, rv_window: int = 30
) -> pd.Series:
    """Vectorized IV/RV ratio series."""
    rv_series = realized_vol(underlying_prices, rv_window)
    return iv_series / rv_series
