"""Bollinger Band Mean Reversion + RSI strategy — long-only 1h swing with score-based quality filter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import BacktestSignalSet, BaseStrategy, Signal
from strategies.registry import register


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _bollinger_bands(
    close: pd.Series, period: int = 20, std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (upper, middle, lower) Bollinger Bands using population std."""
    middle = close.rolling(period).mean()
    rolling_std = close.rolling(period).std(ddof=0)
    upper = middle + std * rolling_std
    lower = middle - std * rolling_std
    return upper, middle, lower


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


@register
class BBMeanReversionStrategy(BaseStrategy):
    """BB mean reversion with RSI oversold confirmation. Long-only swing on 1h.

    Entry: price at/below lower BB + RSI < oversold + quality score >= min_score.
    Exit: close >= middle BB (or upper), time exit (max_holding_bars), or ATR-based SL.

    Scoring (max 6):
      +1 BB Bandwidth — bandwidth in healthy range (not squeeze, not blowout)
      +1 RSI Depth — RSI < deep_oversold threshold
      +1 Volume Spike — volume > vol_mult x SMA(20)
      +1 Daily Uptrend — prev day close > EMA(200)
      +1 Bullish Candle — close > open AND close in upper 50% of range
      +1 %B Below Zero — price below lower band (%B < 0)
    """

    name = "bb_mean_reversion"
    timeframes = ["1h"]

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_deep_oversold: float = 25,
        sl_atr_mult: float = 1.5,
        tp_target: str = "middle_band",
        max_holding_bars: int = 30,
        min_score: int = 3,
        vol_mult: float = 1.3,
        bb_bandwidth_min: float = 0.02,
        bb_bandwidth_max: float = 0.10,
        daily_ema_period: int = 200,
        daily_df: pd.DataFrame | None = None,
    ) -> None:
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_deep_oversold = rsi_deep_oversold
        self.sl_atr_mult = sl_atr_mult
        self.tp_target = tp_target
        self.max_holding_bars = max_holding_bars
        self.min_score = min_score
        self.vol_mult = vol_mult
        self.bb_bandwidth_min = bb_bandwidth_min
        self.bb_bandwidth_max = bb_bandwidth_max
        self.daily_ema_period = daily_ema_period
        self.daily_df = daily_df

    def _compute_score(
        self,
        rsi_val: float,
        bandwidth: float,
        volume_val: float,
        vol_sma_val: float,
        pct_b: float,
        close_val: float,
        open_val: float,
        high_val: float,
        low_val: float,
        daily_uptrend: bool,
    ) -> int:
        """Compute 6-point quality score for a single bar."""
        score = 0
        # 1. BB Bandwidth in healthy range
        if self.bb_bandwidth_min <= bandwidth <= self.bb_bandwidth_max:
            score += 1
        # 2. RSI depth
        if rsi_val < self.rsi_deep_oversold:
            score += 1
        # 3. Volume spike
        if vol_sma_val > 0 and volume_val > self.vol_mult * vol_sma_val:
            score += 1
        # 4. Daily uptrend
        if daily_uptrend:
            score += 1
        # 5. Bullish candle: close > open AND close in upper 50% of range
        candle_range = high_val - low_val
        if candle_range > 0 and close_val > open_val:
            if (close_val - low_val) / candle_range >= 0.5:
                score += 1
        # 6. %B below zero
        if pct_b < 0:
            score += 1
        return score

    def backtest_signals(self, candles: dict[str, pd.DataFrame]) -> BacktestSignalSet:
        df = next(iter(candles.values())).copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        opn = df["open"]
        volume = df["volume"]
        ts = df["timestamp"]

        # --- Indicators ---
        upper, middle, lower = _bollinger_bands(close, self.bb_period, self.bb_std)
        rsi = _rsi(close, self.rsi_period)
        atr = _atr(high, low, close)
        vol_sma = volume.rolling(20).mean()
        bandwidth = (upper - lower) / middle
        pct_b = (close - lower) / (upper - lower)

        # --- Daily EMA(200) trend filter ---
        if self.daily_df is not None:
            daily_close = self.daily_df.set_index("timestamp")["close"]
        else:
            df_ts = df.set_index("timestamp")
            daily_close = df_ts["close"].resample("1D").last().dropna()
        ema200 = _ema(daily_close, self.daily_ema_period)
        daily_above = (daily_close > ema200).shift(1)

        def _is_daily_uptrend(bar_ts: pd.Timestamp) -> bool:
            bar_date = pd.Timestamp(bar_ts.date())
            mask = daily_above.index <= bar_date
            if not mask.any():
                return False
            return bool(daily_above.loc[daily_above.index[mask][-1]])

        # --- Market hours filter (03:45-10:00 UTC) ---
        bar_mins = ts.dt.hour * 60 + ts.dt.minute
        in_market = (bar_mins >= 3 * 60 + 45) & (bar_mins <= 10 * 60)

        # --- Vectorized entry conditions ---
        bb_touch = close <= lower
        rsi_oversold = rsi < self.rsi_oversold

        # Compute scores vectorized
        scores = pd.Series(0, index=df.index)
        for i in df.index:
            if not (bb_touch.loc[i] and rsi_oversold.loc[i] and in_market.loc[i]):
                continue
            bw = bandwidth.loc[i]
            if np.isnan(bw):
                continue
            scores.loc[i] = self._compute_score(
                rsi_val=rsi.loc[i],
                bandwidth=bw,
                volume_val=volume.loc[i],
                vol_sma_val=vol_sma.loc[i] if not np.isnan(vol_sma.loc[i]) else 0,
                pct_b=pct_b.loc[i],
                close_val=close.loc[i],
                open_val=opn.loc[i],
                high_val=high.loc[i],
                low_val=low.loc[i],
                daily_uptrend=_is_daily_uptrend(ts.loc[i]),
            )

        raw_entries = bb_touch & rsi_oversold & (scores >= self.min_score) & in_market
        # Shift entries by 1 to avoid lookahead (enter on next bar's open)
        entries = raw_entries.shift(1).fillna(False).infer_objects(copy=False).astype(bool)

        # --- TP exit: close crosses middle (or upper) band ---
        if self.tp_target == "upper_band":
            tp_condition = close >= upper
        else:
            tp_condition = close >= middle
        raw_exits = tp_condition
        exits = raw_exits.shift(1).fillna(False).infer_objects(copy=False).astype(bool)

        # --- Time-based exit: post-processing loop ---
        # Track bars since entry and force exit at max_holding_bars
        in_position = False
        bars_held = 0
        for i in range(len(df)):
            if entries.iloc[i] and not in_position:
                in_position = True
                bars_held = 0
            elif in_position:
                bars_held += 1
                if bars_held >= self.max_holding_bars:
                    exits.iloc[i] = True
                    in_position = False
                    bars_held = 0
                elif exits.iloc[i]:
                    in_position = False
                    bars_held = 0

        # --- SL stop as pct of price (for vbt) ---
        sl_price = close - self.sl_atr_mult * atr
        sl_stop = ((close - sl_price) / close).clip(lower=0.001)

        return BacktestSignalSet(
            entries=entries,
            exits=exits,
            sl_stop=sl_stop,
            tp_stop=None,
            high=high,
            low=low,
        )

    def on_candle(self, candles: dict[str, pd.DataFrame]) -> Signal | None:
        """Generate live signal from 1h buffer. Stateless recomputation each bar."""
        if not self.validate_candles(candles):
            return None

        df = next(iter(candles.values()))
        min_bars = max(self.bb_period, self.rsi_period) + 5
        if len(df) < min_bars:
            return None

        close = df["close"]
        high = df["high"]
        low = df["low"]
        opn = df["open"]
        volume = df["volume"]
        ts = df["timestamp"]

        # Market hours check on latest bar
        bar_ts = ts.iloc[-1]
        bar_mins = bar_ts.hour * 60 + bar_ts.minute
        if not (3 * 60 + 45 <= bar_mins <= 10 * 60):
            return None

        # --- Indicators on latest bar ---
        upper, middle, lower = _bollinger_bands(close, self.bb_period, self.bb_std)
        rsi = _rsi(close, self.rsi_period)
        atr = _atr(high, low, close)
        vol_sma = volume.rolling(20).mean()
        bandwidth = (upper - lower) / middle
        pct_b = (close - lower) / (upper - lower)

        cur_close = close.iloc[-1]
        cur_lower = lower.iloc[-1]
        cur_rsi = rsi.iloc[-1]

        # Entry conditions
        if cur_close > cur_lower:
            return None
        if cur_rsi >= self.rsi_oversold:
            return None

        # Daily uptrend check
        daily_uptrend = False
        if self.daily_df is not None:
            daily_close = self.daily_df.set_index("timestamp")["close"]
            ema200 = _ema(daily_close, self.daily_ema_period)
            daily_above = (daily_close > ema200).shift(1)
            bar_date = pd.Timestamp(bar_ts.date())
            mask = daily_above.index <= bar_date
            if mask.any():
                daily_uptrend = bool(daily_above.loc[daily_above.index[mask][-1]])

        # Score
        bw = bandwidth.iloc[-1]
        if np.isnan(bw):
            return None

        score = self._compute_score(
            rsi_val=cur_rsi,
            bandwidth=bw,
            volume_val=volume.iloc[-1],
            vol_sma_val=vol_sma.iloc[-1] if not np.isnan(vol_sma.iloc[-1]) else 0,
            pct_b=pct_b.iloc[-1],
            close_val=cur_close,
            open_val=opn.iloc[-1],
            high_val=high.iloc[-1],
            low_val=low.iloc[-1],
            daily_uptrend=daily_uptrend,
        )

        if score < self.min_score:
            return None

        # SL / TP
        cur_atr = atr.iloc[-1]
        sl_price = cur_close - self.sl_atr_mult * cur_atr
        if self.tp_target == "upper_band":
            tp_price = upper.iloc[-1]
        else:
            tp_price = middle.iloc[-1]

        return Signal(
            direction="BUY",
            symbol="",
            strength=score / 6,
            sl_price=round(float(sl_price), 2),
            tp_price=round(float(tp_price), 2),
            score=score,
        )
