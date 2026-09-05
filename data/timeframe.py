"""Aggregate 1-minute candles into higher timeframes."""

from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd


def _parse_tf_minutes(tf: str) -> int:
    """Parse timeframe string like '5m' or '1h' into minutes."""
    if tf.endswith("h"):
        return int(tf.rstrip("h")) * 60
    return int(tf.rstrip("m"))


class TimeframeAggregator:
    """Aggregates 1m candles into multiple higher timeframes with rolling buffers.

    Args:
        timeframes: List of target timeframe strings (e.g. ["5m", "15m"]).
        buffer_size: Max candles to retain per symbol per timeframe.
    """

    def __init__(self, timeframes: list[str], buffer_size: int = 500) -> None:
        self.timeframes = {tf: _parse_tf_minutes(tf) for tf in timeframes}
        self.buffer_size = buffer_size
        # {symbol: {tf: deque of completed candles}}
        self._buffers: dict[str, dict[str, deque]] = defaultdict(
            lambda: {tf: deque(maxlen=buffer_size) for tf in self.timeframes}
        )
        # {symbol: {tf: list of pending 1m candles}}
        self._pending: dict[str, dict[str, list]] = defaultdict(
            lambda: {tf: [] for tf in self.timeframes}
        )

    def add_candle(self, symbol: str, candle_1m: dict) -> None:
        """Ingest a 1m candle and aggregate into all registered timeframes.

        Args:
            symbol: Instrument symbol.
            candle_1m: Dict with keys timestamp, open, high, low, close, volume.
        """
        for tf, n_minutes in self.timeframes.items():
            pending = self._pending[symbol][tf]
            pending.append(candle_1m)

            if len(pending) >= n_minutes:
                agg = {
                    "timestamp": pending[0]["timestamp"],
                    "open": pending[0]["open"],
                    "high": max(c["high"] for c in pending),
                    "low": min(c["low"] for c in pending),
                    "close": pending[-1]["close"],
                    "volume": sum(c["volume"] for c in pending),
                }
                self._buffers[symbol][tf].append(agg)
                pending.clear()

    def get_candles(self, symbol: str, tf: str) -> pd.DataFrame:
        """Return the rolling buffer for a symbol/timeframe as a DataFrame.

        Args:
            symbol: Instrument symbol.
            tf: Timeframe string (e.g. "5m").

        Returns:
            DataFrame with columns [timestamp, open, high, low, close, volume].
        """
        buf = self._buffers.get(symbol, {}).get(tf, deque())
        if not buf:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        return pd.DataFrame(list(buf))
