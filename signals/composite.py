"""Composite vol signal snapshot and VRP entry logic."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from config.options_config import VRPConfig
from data.chain import OptionChain
from signals.flat_forward import flat_forward_signal
from signals.iv_percentile import iv_percentile


@dataclass
class VolSignalSnapshot:
    underlying: str
    timestamp: datetime
    atm_iv: float
    rv_30d: float
    ivrv_ratio: float
    iv_percentile_rank: float
    flat_forward_ratio: float | None
    days_to_expiry: int


def compute_snapshot(
    chain: OptionChain,
    underlying_prices: pd.Series,
    iv_history: pd.Series,
    far_chain: OptionChain | None = None,
    rv_window: int = 30,
) -> VolSignalSnapshot:
    """Build a VolSignalSnapshot from chain and historical data.

    Args:
        chain: Current near-term option chain.
        underlying_prices: Historical underlying close prices.
        iv_history: Historical ATM IV series for percentile calc.
        far_chain: Optional far-term chain for flat-forward signal.
        rv_window: Window for realized vol computation.
    """
    from data.iv_engine import realized_vol

    atm_iv = chain.atm_iv()
    rv_series = realized_vol(underlying_prices, rv_window).dropna()
    if rv_series.empty:
        rv_30d = 0.0
    else:
        rv_30d = float(rv_series.iloc[-1])
    ivrv = atm_iv / rv_30d if rv_30d > 0 else 0.0
    iv_pct = iv_percentile(atm_iv, iv_history)
    ff_ratio = flat_forward_signal(chain, far_chain) if far_chain is not None else None

    expiry_date = pd.Timestamp(chain.expiry)
    dte = (expiry_date - pd.Timestamp(datetime.now().date())).days

    return VolSignalSnapshot(
        underlying=chain.underlying,
        timestamp=datetime.now(),
        atm_iv=atm_iv,
        rv_30d=float(rv_30d),
        ivrv_ratio=ivrv,
        iv_percentile_rank=iv_pct,
        flat_forward_ratio=ff_ratio,
        days_to_expiry=dte,
    )


def is_vrp_entry(snapshot: VolSignalSnapshot, config: VRPConfig) -> bool:
    """True if snapshot meets VRP entry conditions."""
    return (
        snapshot.ivrv_ratio >= config.ivrv_entry_threshold
        and config.iv_percentile_entry_min <= snapshot.iv_percentile_rank <= config.iv_percentile_entry_max
    )
