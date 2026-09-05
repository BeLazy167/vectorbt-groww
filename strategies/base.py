"""Abstract base strategy and Signal type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestSignalSet:
    """Extended backtest output with stop-loss/take-profit support."""

    entries: pd.Series
    exits: pd.Series
    sl_stop: pd.Series | float | None = None
    tp_stop: pd.Series | float | None = None
    high: pd.Series | None = None
    low: pd.Series | None = None


@dataclass(frozen=True)
class Signal:
    direction: str  # "BUY" or "SELL"
    symbol: str
    strength: float  # 0-1
    sl_price: float | None = None
    tp_price: float | None = None
    score: int = 0


class BaseStrategy(ABC):
    """Base class all strategies must extend."""

    name: str
    timeframes: list[str]

    @abstractmethod
    def on_candle(self, candles: dict[str, pd.DataFrame]) -> Signal | None:
        """Generate a live signal from the latest candles.

        Args:
            candles: Maps timeframe string (e.g. "1m") to DataFrame of
                recent OHLCV candles.

        Returns:
            A Signal if conditions are met, else None.
        """

    @abstractmethod
    def backtest_signals(
        self, candles: dict[str, pd.DataFrame]
    ) -> tuple[pd.Series, pd.Series]:
        """Produce boolean entry/exit series for vectorbt backtesting.

        Args:
            candles: Maps timeframe string to full historical DataFrame.

        Returns:
            Tuple of (entries, exits) as boolean pd.Series aligned to 1m index.
        """

    def validate_candles(self, candles: dict[str, pd.DataFrame]) -> bool:
        """Check all required timeframes are present with enough data."""
        for tf in self.timeframes:
            if tf not in candles or candles[tf].empty:
                return False
        return True
