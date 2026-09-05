"""Option chain fetching and parsing from Groww API."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
from growwapi import GrowwAPI

from signals.base import OptionType

logger = logging.getLogger(__name__)


@dataclass
class OptionContract:
    trading_symbol: str
    strike: float
    option_type: OptionType
    ltp: float
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float
    oi: int
    volume: int


@dataclass
class OptionChain:
    underlying: str
    underlying_ltp: float
    expiry: str
    contracts: list[OptionContract]

    def atm_strike(self) -> float:
        strikes = {c.strike for c in self.contracts}
        return min(strikes, key=lambda s: abs(s - self.underlying_ltp))

    def atm_contracts(self) -> tuple[OptionContract, OptionContract]:
        """ATM CE and PE contracts.

        Returns:
            Tuple of (ATM CE, ATM PE).

        Raises:
            ValueError: If ATM CE or PE not found.
        """
        strike = self.atm_strike()
        ce = self.get_contract(strike, OptionType.CE)
        pe = self.get_contract(strike, OptionType.PE)
        if ce is None or pe is None:
            raise ValueError(f"ATM contracts not found for strike {strike}")
        return ce, pe

    def atm_iv(self) -> float:
        ce, pe = self.atm_contracts()
        return (ce.iv + pe.iv) / 2.0

    def straddle_premium(self) -> float:
        ce, pe = self.atm_contracts()
        return ce.ltp + pe.ltp

    def get_contract(self, strike: float, option_type: OptionType) -> OptionContract | None:
        for c in self.contracts:
            if c.strike == strike and c.option_type == option_type:
                return c
        return None


def _load_instruments(groww: GrowwAPI) -> pd.DataFrame:
    """Load instruments with SQLite cache (refreshes daily)."""
    if not hasattr(_load_instruments, "_cache"):
        from data.cache import get_cached_instruments, store_instruments

        cached = get_cached_instruments()
        if cached is not None:
            logger.info("Instruments cache hit (%d rows)", len(cached))
            _load_instruments._cache = cached
        else:
            logger.info("Fetching instruments from API...")
            _load_instruments._cache = groww.get_all_instruments()
            store_instruments(_load_instruments._cache)
            logger.info("Cached %d instruments", len(_load_instruments._cache))
    return _load_instruments._cache


def fetch_option_chain(
    groww: GrowwAPI,
    underlying: str,
    expiry: str,
    exchange: str = "NSE",
) -> OptionChain:
    """Fetch and parse option chain from instruments + quote + greeks APIs.

    Args:
        groww: Authenticated GrowwAPI instance.
        underlying: Underlying symbol (e.g. "NIFTY").
        expiry: Expiry date string (YYYY-MM-DD).
        exchange: Exchange code.

    Returns:
        Parsed OptionChain with all contracts.
    """
    instruments = _load_instruments(groww)
    opts = instruments[
        (instruments["underlying_symbol"] == underlying)
        & (instruments["expiry_date"] == expiry)
        & (instruments["instrument_type"].isin(["CE", "PE"]))
        & (instruments["exchange"] == exchange)
    ]

    if opts.empty:
        raise ValueError(f"No option instruments for {underlying} expiry {expiry}")

    # Get underlying LTP
    underlying_ltp_resp = groww.get_quote(
        trading_symbol=underlying, exchange=exchange, segment="CASH",
    )
    underlying_ltp = underlying_ltp_resp.get("last_price", 0.0)

    # Filter to strikes near ATM (within 10% of underlying) to limit API calls
    strikes = opts["strike_price"].astype(float)
    lower = underlying_ltp * 0.90
    upper = underlying_ltp * 1.10
    near_atm = opts[(strikes >= lower) & (strikes <= upper)]

    contracts: list[OptionContract] = []
    for _, row in near_atm.iterrows():
        sym = row["trading_symbol"]
        strike = float(row["strike_price"])
        ot = OptionType(row["instrument_type"])

        try:
            quote = groww.get_quote(trading_symbol=sym, exchange=exchange, segment="FNO")
        except Exception:
            logger.debug("Quote failed for %s", sym)
            continue

        try:
            greeks_resp = groww.get_greeks(
                exchange=exchange,
                underlying=underlying,
                trading_symbol=sym,
                expiry=expiry,
            )
            g = greeks_resp.get("greeks", {})
        except Exception:
            logger.debug("Greeks failed for %s", sym)
            g = {}

        contracts.append(OptionContract(
            trading_symbol=sym,
            strike=strike,
            option_type=ot,
            ltp=quote.get("last_price", 0.0),
            iv=g.get("iv", 0.0),
            delta=g.get("delta", 0.0),
            gamma=g.get("gamma", 0.0),
            theta=g.get("theta", 0.0),
            vega=g.get("vega", 0.0),
            oi=int(quote.get("open_interest", 0) or 0),
            volume=int(quote.get("volume", 0) or 0),
        ))

    return OptionChain(
        underlying=underlying,
        underlying_ltp=underlying_ltp,
        expiry=expiry,
        contracts=contracts,
    )


def fetch_expiries(
    groww: GrowwAPI,
    underlying: str,
    year: int,
    month: int | None = None,
    exchange: str = "NSE",
) -> list[str]:
    """Get expiry dates from instruments master instead of forbidden API."""
    instruments = _load_instruments(groww)
    opts = instruments[
        (instruments["underlying_symbol"] == underlying)
        & (instruments["instrument_type"].isin(["CE", "PE"]))
        & (instruments["exchange"] == exchange)
    ]
    all_expiries = sorted(opts["expiry_date"].unique())
    # Filter by year
    expiries = [e for e in all_expiries if e.startswith(str(year))]
    if month is not None:
        expiries = [e for e in expiries if int(e.split("-")[1]) == month]
    return expiries
