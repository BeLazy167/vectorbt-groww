"""MA crossover scalper strategy with multi-timeframe confirmation."""

from __future__ import annotations

import pandas as pd

from strategies.base import BaseStrategy, Signal
from strategies.registry import register


@register
class MACrossoverStrategy(BaseStrategy):
    """Fast/slow MA crossover on 1m with 5m trend filter."""

    name = "ma_crossover"
    timeframes = ["1m", "5m"]

    def __init__(self, fast_period: int = 9, slow_period: int = 21) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period

    def on_candle(self, candles: dict[str, pd.DataFrame]) -> Signal | None:
        """Generate live signal from latest 1m/5m candles.

        Args:
            candles: Must contain "1m" and "5m" DataFrames with a "close" column.

        Returns:
            BUY/SELL Signal or None.
        """
        if not self.validate_candles(candles):
            return None

        df_1m = candles["1m"]
        df_5m = candles["5m"]

        fast = df_1m["close"].rolling(self.fast_period).mean()
        slow = df_1m["close"].rolling(self.slow_period).mean()

        # 5m trend filter
        trend_ma = df_5m["close"].rolling(20).mean()
        bullish_5m = df_5m["close"].iloc[-1] > trend_ma.iloc[-1]
        bearish_5m = df_5m["close"].iloc[-1] < trend_ma.iloc[-1]

        # Cross detection on last two bars
        prev_fast, curr_fast = fast.iloc[-2], fast.iloc[-1]
        prev_slow, curr_slow = slow.iloc[-2], slow.iloc[-1]

        cross_above = prev_fast <= prev_slow and curr_fast > curr_slow
        cross_below = prev_fast >= prev_slow and curr_fast < curr_slow

        if cross_above and bullish_5m:
            return Signal(direction="BUY", symbol="", strength=0.7)
        if cross_below and bearish_5m:
            return Signal(direction="SELL", symbol="", strength=0.7)
        return None

    def backtest_signals(
        self, candles: dict[str, pd.DataFrame]
    ) -> tuple[pd.Series, pd.Series]:
        """Compute entry/exit boolean series for vectorbt.

        Args:
            candles: Must contain "1m" DataFrame with a "close" column.

        Returns:
            (entries, exits) boolean Series on 1m index, shifted to avoid lookahead.
        """
        # Use whatever interval is available (1m, 5m, etc.)
        df = next(iter(candles.values()))
        close = df["close"]

        fast = close.rolling(self.fast_period).mean()
        slow = close.rolling(self.slow_period).mean()

        entries = ((fast > slow) & (fast.shift(1) <= slow.shift(1))).shift(1, fill_value=False)
        exits = ((fast < slow) & (fast.shift(1) >= slow.shift(1))).shift(1, fill_value=False)

        return entries, exits
