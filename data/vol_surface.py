"""Volatility surface analytics: vol cone, term structure, forward ratio."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from data.chain import OptionChain


def vol_cone(
    iv_series: pd.Series,
    percentiles: list[int] | None = None,
) -> pd.DataFrame:
    """Rolling percentile bands of IV series.

    Args:
        iv_series: Historical IV series indexed by date.
        percentiles: Percentile levels to compute.

    Returns:
        DataFrame with columns for each percentile band.
    """
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90]
    result = {}
    for p in percentiles:
        result[f"p{p}"] = iv_series.rolling(len(iv_series), min_periods=1).quantile(p / 100.0)
    return pd.DataFrame(result, index=iv_series.index)


def term_structure(chains: list[OptionChain]) -> pd.DataFrame:
    today = datetime.now().date()
    rows = []
    for chain in chains:
        expiry_date = pd.Timestamp(chain.expiry).date()
        dte = (expiry_date - today).days
        rows.append({
            "expiry": chain.expiry,
            "atm_iv": chain.atm_iv(),
            "days_to_expiry": dte,
        })
    df = pd.DataFrame(rows)
    return df.sort_values("days_to_expiry").reset_index(drop=True)


def flat_forward_ratio(near_chain: OptionChain, far_chain: OptionChain) -> float:
    return near_chain.atm_iv() / far_chain.atm_iv()
