"""Backtest engine powered by vectorbt."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import vectorbt as vbt
from growwapi import GrowwAPI

from config.settings import Settings
from data.historical import fetch_historical
from data.timeframe import TimeframeAggregator
from strategies.base import BacktestSignalSet
from strategies.registry import get as get_strategy


@dataclass
class BacktestResult:
    portfolio: vbt.Portfolio
    strategy_name: str
    symbol: str
    start: datetime
    end: datetime


class BacktestEngine:
    """Runs vectorbt backtests using historical Groww data.

    Args:
        settings: Global pipeline settings.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(
        self, strategy_name: str, symbol: str, start: datetime, end: datetime,
        interval: int = 1,
    ) -> BacktestResult:
        """Execute a full backtest for a strategy over a date range.

        Args:
            strategy_name: Registered strategy name (see strategies.registry).
            symbol: Trading symbol (e.g. "RELIANCE").
            start: Backtest start datetime.
            end: Backtest end datetime.
            interval: Candle interval in minutes (1, 5, 15, etc.).

        Returns:
            BacktestResult containing the vbt Portfolio and run metadata.
        """
        token = self.settings.groww.get_token()
        groww = GrowwAPI(token)

        df_base = fetch_historical(
            groww,
            symbol,
            start,
            end,
            interval=interval,
            exchange=self.settings.groww.exchange,
            segment=self.settings.groww.segment,
        )

        strategy_cls = get_strategy(strategy_name)
        strategy = strategy_cls()

        tf_key = f"{interval}m"
        candles: dict[str, pd.DataFrame] = {tf_key: df_base}

        # If fetching 1m, also build higher TF candles
        if interval == 1:
            agg = TimeframeAggregator(
                strategy.timeframes, self.settings.trading.candle_buffer_size
            )
            for _, row in df_base.iterrows():
                agg.add_candle(symbol, row.to_dict())
            for tf in strategy.timeframes:
                candles[tf] = agg.get_candles(symbol, tf)
            candles["1m"] = df_base

        result = strategy.backtest_signals(candles)
        close = df_base["close"]

        if isinstance(result, BacktestSignalSet):
            entries, exits = result.entries, result.exits
            stop_kwargs: dict = {}
            if result.sl_stop is not None:
                stop_kwargs["sl_stop"] = result.sl_stop
            if result.tp_stop is not None:
                stop_kwargs["tp_stop"] = result.tp_stop
            if result.high is not None:
                stop_kwargs["high"] = result.high
            if result.low is not None:
                stop_kwargs["low"] = result.low
        else:
            entries, exits = result
            stop_kwargs = {}

        pct_fee, fixed_fee = self.settings.trading.fees.effective_fees(
            self.settings.trading.capital
        )
        pf = vbt.Portfolio.from_signals(
            close,
            entries,
            exits,
            init_cash=self.settings.trading.capital,
            fees=pct_fee,
            fixed_fees=fixed_fee,
            slippage=self.settings.safeguards.slippage_buffer_pct,
            **stop_kwargs,
        )

        return BacktestResult(
            portfolio=pf,
            strategy_name=strategy_name,
            symbol=symbol,
            start=start,
            end=end,
        )
