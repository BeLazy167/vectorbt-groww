"""Multi-indicator confluence strategy — long-only with 6 entry conditions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import BacktestSignalSet, BaseStrategy, Signal
from strategies.registry import register


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average Directional Index."""
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # Zero out where the other is larger
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    atr_vals = _atr(high, low, close, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_vals)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_vals)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _plus_di(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0

    atr_vals = _atr(high, low, close, period)
    return 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_vals)


def _minus_di(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    minus_dm[minus_dm < plus_dm] = 0

    atr_vals = _atr(high, low, close, period)
    return 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_vals)


def _resample_to_15m(df_5m: pd.DataFrame) -> pd.DataFrame:
    """Resample 5m OHLCV to 15m. Handles both DatetimeIndex and 'timestamp' column."""
    if not isinstance(df_5m.index, pd.DatetimeIndex):
        df_5m = df_5m.set_index("timestamp")
    return df_5m.resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()


@register
class MultiConfluenceStrategy(BaseStrategy):
    """Long-only 6-condition confluence on 5m with 15m trend filter.

    Entry requires ALL of:
        1. EMA(fast) crosses above EMA(slow)
        2. MACD histogram > 0
        3. RSI between rsi_low and rsi_high
        4. ADX > adx_threshold AND +DI > -DI
        5. Volume > vol_mult * SMA(vol_period)
        6. 15m close > 15m EMA(trend_period)

    Exit on ANY of:
        - RSI > rsi_exit
        - MACD histogram turns negative
        - ATR-based SL/TP hit
    """

    name = "multi_confluence"
    timeframes = ["5m"]

    def __init__(
        self,
        fast_ema: int = 9,
        slow_ema: int = 21,
        rsi_period: int = 14,
        rsi_low: float = 35,
        rsi_high: float = 75,
        rsi_exit: float = 80,
        adx_period: int = 14,
        adx_threshold: float = 15,
        vol_mult: float = 1.2,
        vol_period: int = 20,
        trend_period: int = 20,
        atr_period: int = 14,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 3.0,
    ) -> None:
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.rsi_period = rsi_period
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.rsi_exit = rsi_exit
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.vol_mult = vol_mult
        self.vol_period = vol_period
        self.trend_period = trend_period
        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult

    def backtest_signals(
        self, candles: dict[str, pd.DataFrame]
    ) -> BacktestSignalSet:
        """Produce entry/exit signals with ATR-based stops.

        Args:
            candles: Must contain a 5m DataFrame with OHLCV columns.

        Returns:
            BacktestSignalSet with entries, exits, sl_stop, tp_stop, high, low.
        """
        df = next(iter(candles.values()))
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # --- Indicators ---
        ema_fast = _ema(close, self.fast_ema)
        ema_slow = _ema(close, self.slow_ema)
        rsi = _rsi(close, self.rsi_period)
        _, _, macd_hist = _macd(close)
        adx = _adx(high, low, close, self.adx_period)
        pdi = _plus_di(high, low, close, self.adx_period)
        mdi = _minus_di(high, low, close, self.adx_period)
        atr = _atr(high, low, close, self.atr_period)
        vol_sma = volume.rolling(self.vol_period).mean()

        # --- 15m trend filter ---
        df_15m = _resample_to_15m(df)
        ema_15m = _ema(df_15m["close"], self.trend_period)
        trend_bullish_15m = df_15m["close"] > ema_15m
        # Forward-fill onto 5m timestamps, then align back to original index
        ts_index = df["timestamp"] if "timestamp" in df.columns else df.index
        trend_bullish = trend_bullish_15m.reindex(ts_index, method="ffill").fillna(False)
        trend_bullish.index = df.index

        # --- Entry conditions (all must be true) ---
        c1_cross = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        c2_macd = macd_hist > 0
        c3_rsi = (rsi >= self.rsi_low) & (rsi <= self.rsi_high)
        c4_adx = (adx > self.adx_threshold) & (pdi > mdi)
        c5_vol = volume > (self.vol_mult * vol_sma)
        c6_trend = trend_bullish

        entries = (c1_cross & c2_macd & c3_rsi & c4_adx & c5_vol & c6_trend).shift(
            1, fill_value=False
        )

        # --- Exit conditions (any triggers) ---
        exit_rsi = rsi > self.rsi_exit
        exit_macd = (macd_hist < 0) & (macd_hist.shift(1) >= 0)
        exits = (exit_rsi | exit_macd).shift(1, fill_value=False)

        # --- ATR stops (per-bar, as fraction of close) ---
        sl_stop = (self.sl_atr_mult * atr) / close
        tp_stop = (self.tp_atr_mult * atr) / close

        return BacktestSignalSet(
            entries=entries,
            exits=exits,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            high=high,
            low=low,
        )

    def on_candle(self, candles: dict[str, pd.DataFrame]) -> Signal | None:
        """Generate live signal from latest candles.

        Args:
            candles: Must contain a 5m DataFrame with OHLCV columns.

        Returns:
            BUY Signal or None (long-only).
        """
        if not self.validate_candles(candles):
            return None

        df = next(iter(candles.values()))
        if len(df) < max(self.slow_ema, 26, self.vol_period) + 5:
            return None

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        ema_fast = _ema(close, self.fast_ema)
        ema_slow = _ema(close, self.slow_ema)

        cross_above = ema_fast.iloc[-1] > ema_slow.iloc[-1] and ema_fast.iloc[-2] <= ema_slow.iloc[-2]
        if not cross_above:
            return None

        _, _, macd_hist = _macd(close)
        if macd_hist.iloc[-1] <= 0:
            return None

        rsi = _rsi(close, self.rsi_period)
        if not (self.rsi_low <= rsi.iloc[-1] <= self.rsi_high):
            return None

        adx = _adx(high, low, close, self.adx_period)
        pdi = _plus_di(high, low, close, self.adx_period)
        mdi = _minus_di(high, low, close, self.adx_period)
        if adx.iloc[-1] <= self.adx_threshold or pdi.iloc[-1] <= mdi.iloc[-1]:
            return None

        vol_sma = volume.rolling(self.vol_period).mean()
        if volume.iloc[-1] <= self.vol_mult * vol_sma.iloc[-1]:
            return None

        # 15m trend check
        df_15m = _resample_to_15m(df)
        if len(df_15m) < self.trend_period:
            return None
        ema_15m = _ema(df_15m["close"], self.trend_period)
        if df_15m["close"].iloc[-1] <= ema_15m.iloc[-1]:
            return None

        return Signal(direction="BUY", symbol="", strength=0.9)
