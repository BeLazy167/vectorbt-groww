"""F&O universe screener: filter underlyings by liquidity thresholds."""

from __future__ import annotations

from datetime import datetime

from growwapi import GrowwAPI

from data.chain import fetch_expiries, fetch_option_chain


def screen_universe(
    groww: GrowwAPI,
    underlyings: list[str],
    min_volume: int = 1000,
    min_oi: int = 5000,
    exchange: str = "NSE",
) -> list[str]:
    """Filter underlyings by ATM option liquidity.

    Args:
        groww: Authenticated GrowwAPI instance.
        underlyings: Candidate underlying symbols.
        min_volume: Minimum ATM contract volume.
        min_oi: Minimum ATM contract open interest.
        exchange: Exchange code.

    Returns:
        List of underlyings passing liquidity thresholds.
    """
    now = datetime.now()
    qualified: list[str] = []

    for sym in underlyings:
        try:
            expiries = fetch_expiries(groww, sym, year=now.year, month=now.month, exchange=exchange)
            if not expiries:
                continue
            nearest_expiry = expiries[0]
            chain = fetch_option_chain(groww, sym, nearest_expiry, exchange=exchange)
            ce, pe = chain.atm_contracts()
            if (
                ce.volume >= min_volume
                and pe.volume >= min_volume
                and ce.oi >= min_oi
                and pe.oi >= min_oi
            ):
                qualified.append(sym)
        except (ValueError, KeyError):
            continue

    return qualified
