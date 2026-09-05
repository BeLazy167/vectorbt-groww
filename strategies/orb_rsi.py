"""Opening Range Breakout + RSI strategy — long-only intraday with score-based fakeout filter."""

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


def _vwap_daily(df: pd.DataFrame) -> pd.Series:
    """Compute intraday VWAP, resetting each day."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    dates = df["timestamp"].dt.date
    vwap = pd.Series(np.nan, index=df.index)
    for _, grp in df.groupby(dates):
        idx = grp.index
        tp = typical.loc[idx]
        vol = df["volume"].loc[idx]
        cum_tp_vol = (tp * vol).cumsum()
        cum_vol = vol.cumsum().replace(0, np.nan)
        vwap.loc[idx] = cum_tp_vol / cum_vol
    return vwap


def _compute_orb(
    df: pd.DataFrame, orb_minutes: int = 15, market_open_hour: int = 3, market_open_min: int = 45,
) -> tuple[pd.Series, pd.Series]:
    """Compute the Opening Range High/Low for each trading day."""
    ts = df["timestamp"]
    dates = ts.dt.date

    orb_high = pd.Series(np.nan, index=df.index)
    orb_low = pd.Series(np.nan, index=df.index)

    for date, grp in df.groupby(dates):
        mask = (grp["timestamp"].dt.hour == market_open_hour) & (
            grp["timestamp"].dt.minute < (market_open_min + orb_minutes)
        )
        orb_candles = grp[mask]
        if orb_candles.empty:
            continue
        h = orb_candles["high"].max()
        l = orb_candles["low"].min()
        orb_high.loc[grp.index] = h
        orb_low.loc[grp.index] = l

    return orb_high, orb_low


def _is_after_orb(
    ts: pd.Series, orb_minutes: int = 15, market_open_hour: int = 3, market_open_min: int = 45,
) -> pd.Series:
    """Boolean mask: True for bars after the opening range window."""
    cutoff_total_min = market_open_hour * 60 + market_open_min + orb_minutes
    bar_total_min = ts.dt.hour * 60 + ts.dt.minute
    return bar_total_min >= cutoff_total_min


@register
class ORBRSIStrategy(BaseStrategy):
    """ORB + RSI with score-based fakeout filter + quality gates. Long-only intraday on 5m.

    Scoring (max 6):
      +1 Volume — breakout candle volume > vol_mult x SMA(20)
      +1 Candle body — body > body_pct of candle range (strong close, not wick)
      +1 Consecutive closes — 2+ consecutive 5m closes above ORB high before retest
      +1 ORB range — ORB range between orb_range_min% and orb_range_max% of open
      +1 VWAP — retest candle close > VWAP
      +1 Time — entry before time_cutoff UTC (12:30 IST = 07:00 UTC)

    Hard gates (all must pass in addition to score >= min_score):
      - Follow-through: max high between BO and retest > min_follow_through x ORB range above ORB high
      - Post-BO momentum: 5 bars after BO must show positive return
      - Turnover: trailing 15m avg turnover >= min_turnover_cr crores

    Entry requires score >= min_score AND all hard gates pass.
    """

    name = "orb_rsi"
    timeframes = ["5m"]

    def __init__(
        self,
        orb_minutes: int = 15,
        rsi_period: int = 14,
        rsi_threshold: float = 40,
        rsi_timeframe: str = "1h",
        atr_period: int = 14,
        fib_mult: float = 1.618,
        min_score: int = 4,
        vol_mult: float = 1.5,
        body_pct: float = 0.6,
        orb_range_min: float = 0.003,
        orb_range_max: float = 0.015,
        time_cutoff_hour: int = 7,
        time_cutoff_min: int = 0,
        ema_trend_period: int = 200,
        use_ema_filter: bool = True,
        daily_df: pd.DataFrame | None = None,
        sl_atr_mult: float = 1.5,
        min_follow_through: float = 0.3,
        require_positive_momentum: bool = True,
        bo_momentum_bars: int = 5,
        min_turnover_cr: float = 0.0,
    ) -> None:
        self.orb_minutes = orb_minutes
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
        self.rsi_timeframe = rsi_timeframe
        self.atr_period = atr_period
        self.fib_mult = fib_mult
        self.min_score = min_score
        self.vol_mult = vol_mult
        self.body_pct = body_pct
        self.orb_range_min = orb_range_min
        self.orb_range_max = orb_range_max
        self.time_cutoff_hour = time_cutoff_hour
        self.time_cutoff_min = time_cutoff_min
        self.ema_trend_period = ema_trend_period
        self.use_ema_filter = use_ema_filter
        self.daily_df = daily_df
        self.sl_atr_mult = sl_atr_mult
        self.min_follow_through = min_follow_through
        self.require_positive_momentum = require_positive_momentum
        self.bo_momentum_bars = bo_momentum_bars
        self.min_turnover_cr = min_turnover_cr

    def backtest_signals(
        self, candles: dict[str, pd.DataFrame]
    ) -> BacktestSignalSet:
        df = next(iter(candles.values())).copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        opn = df["open"]
        volume = df["volume"]

        # --- Higher TF RSI ---
        df_ts = df.set_index("timestamp")
        df_htf = df_ts.resample(self.rsi_timeframe).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        rsi_htf = _rsi(df_htf["close"], self.rsi_period)
        rsi_htf_prev = rsi_htf.shift(1)
        rsi_on_5m = rsi_htf_prev.reindex(df["timestamp"], method="ffill").values
        rsi_series = pd.Series(rsi_on_5m, index=df.index)

        # --- ORB ---
        orb_high, orb_low = _compute_orb(df, self.orb_minutes)
        after_orb = _is_after_orb(df["timestamp"], self.orb_minutes)

        # --- ATR ---
        atr = _atr(high, low, close, self.atr_period)

        # --- VWAP ---
        vwap = _vwap_daily(df)

        # --- Volume SMA(20) ---
        vol_sma = volume.rolling(20).mean()

        # --- Trailing 15m avg turnover (for quality gate) ---
        turnover_5m = close * volume
        turnover_15m = turnover_5m.rolling(3).sum()  # 3x5m = 15m
        avg_turnover_15m = turnover_15m.rolling(60).mean()  # ~5 trading days of 15m bars

        # --- Pre-compute per-bar helpers ---
        candle_range = (high - low).replace(0, np.nan)
        body = (close - opn).abs()
        body_ratio = body / candle_range

        # --- ORB range as pct of day open ---
        orb_range = orb_high - orb_low
        dates = df["timestamp"].dt.date
        day_open = pd.Series(np.nan, index=df.index)
        for _, grp in df.groupby(dates):
            day_open.loc[grp.index] = opn.loc[grp.index.min()]
        orb_range_pct = orb_range / day_open

        # --- Consecutive closes above ORB high ---
        above_orb = (close > orb_high).astype(int)
        consec_above = above_orb.copy()
        for i in range(1, len(consec_above)):
            if consec_above.iloc[i] == 1:
                consec_above.iloc[i] = consec_above.iloc[i - 1] + 1
            else:
                consec_above.iloc[i] = 0

        # --- RSI gate ---
        rsi_ok = rsi_series < self.rsi_threshold

        # --- Breakout: close > ORB high + RSI ok + after ORB ---
        breakout_bar = (close > orb_high) & rsi_ok & after_orb

        # --- Score-based retest entry per day ---
        entries = pd.Series(False, index=df.index)
        entry_scores = pd.Series(0, index=df.index)

        for date, grp in df.groupby(dates):
            day_idx = grp.index
            bo_today = breakout_bar.loc[day_idx]
            bo_bars = bo_today[bo_today]
            if bo_bars.empty:
                continue

            bo_first = bo_bars.index[0]
            orb_h = orb_high.loc[bo_first]
            orb_l = orb_low.loc[bo_first]
            orb_rng = orb_h - orb_l if not np.isnan(orb_h) and not np.isnan(orb_l) else 0
            after_bo = day_idx[day_idx > bo_first]

            # Count consecutive closes above ORB high at breakout point
            consec_at_bo = consec_above.loc[bo_first]

            # Pre-compute post-BO momentum (5 bars after breakout)
            bo_pos = day_idx.get_loc(bo_first) if hasattr(day_idx, 'get_loc') else list(day_idx).index(bo_first)
            momentum_end = min(bo_pos + self.bo_momentum_bars + 1, len(day_idx))
            momentum_slice = close.loc[day_idx[bo_pos:momentum_end]]
            if len(momentum_slice) >= 2:
                bo_momentum = (momentum_slice.iloc[-1] - momentum_slice.iloc[0]) / momentum_slice.iloc[0]
            else:
                bo_momentum = 0.0

            for idx in after_bo:
                # Retest condition: low touches ORB high, close holds above
                if low.loc[idx] <= orb_h and close.loc[idx] >= orb_h:

                    # --- Hard gate: follow-through ---
                    if orb_rng > 0 and self.min_follow_through > 0:
                        max_high_since_bo = high.loc[bo_first:idx].max()
                        follow_through = (max_high_since_bo - orb_h) / orb_rng
                        if follow_through < self.min_follow_through:
                            break  # skip this day

                    # --- Hard gate: post-BO momentum ---
                    if self.require_positive_momentum and bo_momentum <= 0:
                        break  # skip this day

                    # --- Hard gate: turnover ---
                    if self.min_turnover_cr > 0:
                        to_val = avg_turnover_15m.loc[idx]
                        if np.isnan(to_val) or to_val < self.min_turnover_cr * 1e7:
                            break  # skip this day

                    score = 0

                    # 1. Volume on breakout candle
                    if vol_sma.loc[bo_first] > 0 and volume.loc[bo_first] > self.vol_mult * vol_sma.loc[bo_first]:
                        score += 1

                    # 2. Breakout candle body strength
                    if body_ratio.loc[bo_first] >= self.body_pct:
                        score += 1

                    # 3. Consecutive closes above ORB high (at least 2 bars before retest)
                    if consec_at_bo >= 2:
                        score += 1

                    # 4. ORB range in healthy zone
                    rp = orb_range_pct.loc[idx]
                    if not np.isnan(rp) and self.orb_range_min <= rp <= self.orb_range_max:
                        score += 1

                    # 5. VWAP — retest close above VWAP
                    if close.loc[idx] > vwap.loc[idx]:
                        score += 1

                    # 6. Time — entry before cutoff
                    bar_ts = df["timestamp"].loc[idx]
                    bar_mins = bar_ts.hour * 60 + bar_ts.minute
                    cutoff_mins = self.time_cutoff_hour * 60 + self.time_cutoff_min
                    if bar_mins < cutoff_mins:
                        score += 1

                    entry_scores.loc[idx] = score
                    if score >= self.min_score:
                        entries.loc[idx] = True
                    break  # only first retest per day

        # --- EMA(200) daily trend filter ---
        if self.use_ema_filter:
            if self.daily_df is not None:
                daily_close = self.daily_df.set_index("timestamp")["close"]
            else:
                # Build daily from 5m
                daily_close = df_ts["close"].resample("1D").last().dropna()
            ema200 = daily_close.ewm(span=self.ema_trend_period, adjust=False).mean()
            # Use previous day's signal to avoid lookahead
            daily_above = (daily_close > ema200).shift(1)
            for idx in entries[entries].index:
                bar_date = pd.Timestamp(df["timestamp"].iloc[idx].date())
                mask = daily_above.index <= bar_date
                if mask.any() and not daily_above.loc[daily_above.index[mask][-1]]:
                    entries.iloc[idx] = False

        # --- No time-based exit — rely on stops ---
        exits = pd.Series(False, index=df.index)

        # --- Stops ---
        orb_range_abs = orb_high - orb_low
        sl_price = orb_low - self.sl_atr_mult * atr
        tp_price = orb_high + self.fib_mult * orb_range_abs
        sl_stop = ((close - sl_price) / close).clip(lower=0.001)
        tp_stop = ((tp_price - close) / close).clip(lower=0.001)

        return BacktestSignalSet(
            entries=entries,
            exits=exits,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            high=high,
            low=low,
        )

    def on_candle(self, candles: dict[str, pd.DataFrame]) -> Signal | None:
        """Generate live signal with full backtest parity: ORB breakout → hard gates → retest → score → SL/TP."""
        if not self.validate_candles(candles):
            return None

        df = next(iter(candles.values()))
        if len(df) < self.rsi_period + 5:
            return None

        ts = df["timestamp"]
        close = df["close"]
        high = df["high"]
        low = df["low"]
        opn = df["open"]
        volume = df["volume"]

        # Must be after ORB window
        if not _is_after_orb(ts, self.orb_minutes).iloc[-1]:
            return None

        # Filter to today
        today = ts.iloc[-1].date()
        today_mask = ts.dt.date == today
        today_df = df[today_mask].copy()
        today_idx = today_df.index
        if len(today_idx) < 2:
            return None

        # --- ORB ---
        orb_high_s, orb_low_s = _compute_orb(today_df, self.orb_minutes)
        if orb_high_s.isna().all():
            return None
        orb_h = orb_high_s.dropna().iloc[0]
        orb_l = orb_low_s.dropna().iloc[0]
        orb_rng = orb_h - orb_l if not np.isnan(orb_h) and not np.isnan(orb_l) else 0

        # --- Higher TF RSI (previous bar — no lookahead) ---
        df_ts = df.set_index("timestamp")
        df_htf = df_ts.resample(self.rsi_timeframe).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        rsi_htf = _rsi(df_htf["close"], self.rsi_period)
        rsi_htf_prev = rsi_htf.shift(1)
        rsi_on_5m = rsi_htf_prev.reindex(df["timestamp"], method="ffill").values
        rsi_series = pd.Series(rsi_on_5m, index=df.index)

        # --- After ORB + RSI gate for today ---
        after_orb = _is_after_orb(today_df["timestamp"], self.orb_minutes)
        rsi_ok_today = rsi_series.loc[today_idx] < self.rsi_threshold

        # --- First breakout bar today ---
        breakout_bar = (close.loc[today_idx] > orb_h) & rsi_ok_today & after_orb
        bo_bars = breakout_bar[breakout_bar]
        if bo_bars.empty:
            return None
        bo_first = bo_bars.index[0]

        cur = today_idx[-1]
        if cur <= bo_first:
            return None

        # === Hard gate: follow-through ===
        if orb_rng > 0 and self.min_follow_through > 0:
            max_high_since_bo = high.loc[bo_first:cur].max()
            follow_through = (max_high_since_bo - orb_h) / orb_rng
            if follow_through < self.min_follow_through:
                return None

        # === Hard gate: post-BO momentum ===
        if self.require_positive_momentum:
            bo_pos = list(today_idx).index(bo_first)
            momentum_end = min(bo_pos + self.bo_momentum_bars + 1, len(today_idx))
            momentum_slice = close.loc[today_idx[bo_pos:momentum_end]]
            if len(momentum_slice) >= 2:
                bo_momentum = (momentum_slice.iloc[-1] - momentum_slice.iloc[0]) / momentum_slice.iloc[0]
            else:
                bo_momentum = 0.0
            if bo_momentum <= 0:
                return None

        # --- Retest check: current bar low <= ORB high AND close >= ORB high ---
        if not (low.iloc[-1] <= orb_h and close.iloc[-1] >= orb_h):
            return None

        # --- One retest per day: no prior retest between BO and current ---
        between_bo_cur = today_idx[(today_idx > bo_first) & (today_idx < cur)]
        for idx in between_bo_cur:
            if low.loc[idx] <= orb_h and close.loc[idx] >= orb_h:
                return None

        # === 6-Point Score ===
        score = 0
        vol_sma = volume.rolling(20).mean()

        # 1. Volume on breakout candle
        if vol_sma.loc[bo_first] > 0 and volume.loc[bo_first] > self.vol_mult * vol_sma.loc[bo_first]:
            score += 1

        # 2. Breakout candle body strength
        candle_range_bo = high.loc[bo_first] - low.loc[bo_first]
        body_bo = abs(close.loc[bo_first] - opn.loc[bo_first])
        if candle_range_bo > 0 and (body_bo / candle_range_bo) >= self.body_pct:
            score += 1

        # 3. Consecutive closes above ORB high at BO time
        above_orb = (close.loc[today_idx] > orb_h).astype(int)
        consec_above = above_orb.copy()
        for i in range(1, len(consec_above)):
            if consec_above.iloc[i] == 1:
                consec_above.iloc[i] = consec_above.iloc[i - 1] + 1
            else:
                consec_above.iloc[i] = 0
        if consec_above.loc[bo_first] >= 2:
            score += 1

        # 4. ORB range in healthy zone
        day_open_price = opn.loc[today_idx.min()]
        if day_open_price > 0:
            orb_range_pct = orb_rng / day_open_price
            if self.orb_range_min <= orb_range_pct <= self.orb_range_max:
                score += 1

        # 5. VWAP — retest close > intraday VWAP
        vwap = _vwap_daily(today_df)
        if today_df["close"].iloc[-1] > vwap.iloc[-1]:
            score += 1

        # 6. Time — entry before cutoff
        bar_ts = ts.iloc[-1]
        bar_mins = bar_ts.hour * 60 + bar_ts.minute
        cutoff_mins = self.time_cutoff_hour * 60 + self.time_cutoff_min
        if bar_mins < cutoff_mins:
            score += 1

        if score < self.min_score:
            return None

        # === EMA(200) daily trend filter ===
        if self.use_ema_filter and self.daily_df is not None:
            daily_close = self.daily_df.set_index("timestamp")["close"]
            ema200 = daily_close.ewm(span=self.ema_trend_period, adjust=False).mean()
            daily_above = (daily_close > ema200).shift(1)
            bar_date = pd.Timestamp(today)
            mask = daily_above.index <= bar_date
            if mask.any() and not daily_above.loc[daily_above.index[mask][-1]]:
                return None

        # --- Compute SL / TP ---
        atr = _atr(high, low, close, self.atr_period)
        sl_price = orb_l - self.sl_atr_mult * atr.iloc[-1]
        tp_price = orb_h + self.fib_mult * orb_rng

        return Signal(
            direction="BUY",
            symbol="",
            strength=score / 6,
            sl_price=round(float(sl_price), 2),
            tp_price=round(float(tp_price), 2),
            score=score,
        )
