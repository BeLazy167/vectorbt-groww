"""Base signal types for options trading."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OptionType(str, Enum):
    CE = "CE"
    PE = "PE"


class StrategyAction(str, Enum):
    SELL = "SELL"
    BUY = "BUY"


@dataclass(frozen=True)
class OptionLeg:
    """Single leg of a multi-leg options position."""

    trading_symbol: str
    strike: float
    option_type: OptionType
    action: StrategyAction
    lots: int = 1


@dataclass(frozen=True)
class OptionsSignal:
    """Signal output from an options strategy."""

    underlying: str
    expiry: str
    legs: tuple[OptionLeg, ...]
    strategy_name: str
    signal_strength: float  # 0-1
    entry_iv: float
    entry_ivrv: float
    model_score: float | None = None
