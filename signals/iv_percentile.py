"""IV percentile rank computation."""

from __future__ import annotations

import pandas as pd


def iv_percentile(current_iv: float, iv_history: pd.Series) -> float:
    """Rank of current IV vs history, 0-100."""
    if len(iv_history) == 0:
        return 50.0
    return (iv_history < current_iv).sum() / len(iv_history) * 100


def rolling_iv_percentile(iv_series: pd.Series, lookback: int = 252) -> pd.Series:
    """Rolling percentile rank of IV over lookback window."""
    def _pct_rank(window: pd.Series) -> float:
        return (window.iloc[:-1] < window.iloc[-1]).sum() / (len(window) - 1) * 100

    return iv_series.rolling(lookback, min_periods=lookback).apply(_pct_rank, raw=False)
