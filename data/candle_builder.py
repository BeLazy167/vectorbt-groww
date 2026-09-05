"""Build 1-minute OHLCV candles from LTP tick stream."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Callable


class CandleBuilder:
    """Accumulates LTP ticks into 1-minute OHLCV candles.

    Emits a completed candle via callback when a tick arrives
    in the next minute boundary.

    Args:
        on_candle_complete: Callback invoked with (symbol, candle_dict)
            when a 1m candle is finalized.
    """

    def __init__(self, on_candle_complete: Callable[[str, dict], None]) -> None:
        self.on_candle_complete = on_candle_complete
        self._current: dict[str, dict] = {}

    def on_tick(self, symbol: str, ltp: float, timestamp: datetime) -> None:
        """Process an incoming LTP tick.

        Args:
            symbol: Instrument symbol.
            ltp: Last traded price.
            timestamp: Tick timestamp.
        """
        minute_key = timestamp.replace(second=0, microsecond=0)
        candle = self._current.get(symbol)

        if candle is not None and candle["timestamp"] != minute_key:
            self.on_candle_complete(symbol, candle)
            candle = None

        if candle is None:
            self._current[symbol] = {
                "timestamp": minute_key,
                "open": ltp,
                "high": ltp,
                "low": ltp,
                "close": ltp,
                "volume": 1,
            }
        else:
            candle["high"] = max(candle["high"], ltp)
            candle["low"] = min(candle["low"], ltp)
            candle["close"] = ltp
            candle["volume"] += 1

    def flush(self, symbol: str) -> None:
        """Force-emit the current in-progress candle for a symbol."""
        candle = self._current.pop(symbol, None)
        if candle is not None:
            self.on_candle_complete(symbol, candle)

    def flush_all(self) -> None:
        """Force-emit all in-progress candles."""
        for symbol in list(self._current):
            self.flush(symbol)
