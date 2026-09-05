"""Async live trading runner."""

from __future__ import annotations

import dataclasses
import logging
from datetime import datetime, timedelta

import pandas as pd
from growwapi import GrowwAPI, GrowwFeed

from config.settings import Settings
from data.candle_builder import CandleBuilder
from data.feed import LiveFeed
from data.historical import fetch_historical
from data.timeframe import TimeframeAggregator
from execution.broker import Broker
from execution.position_manager import PositionManager
from execution.safeguards import SafeguardChecker
from strategies.base import BaseStrategy, Signal
from strategies.registry import get as get_strategy

logger = logging.getLogger(__name__)


class LiveRunner:
    """Orchestrates live trading: feed → strategy → safeguards → broker.

    Args:
        settings: Pipeline configuration.
        strategy_name: Registered strategy name.
        symbols: List of trading symbols.
        instruments: List of instrument dicts for GrowwFeed subscription.
    """

    def __init__(
        self,
        settings: Settings,
        strategy_name: str,
        symbols: list[str],
        instruments: list[dict],
    ) -> None:
        self.settings = settings
        self.symbols = symbols
        self.instruments = instruments

        # Auth
        token = settings.groww.get_token()
        self.groww = GrowwAPI(token)

        # Strategy
        strategy_cls = get_strategy(strategy_name)
        self.strategy: BaseStrategy = strategy_cls()

        # Data pipeline
        higher_tfs = [tf for tf in self.strategy.timeframes if tf != "1m"]
        self.aggregator = TimeframeAggregator(
            higher_tfs, settings.trading.candle_buffer_size
        )
        self.candle_builder = CandleBuilder(on_candle_complete=self._on_candle)
        self.feed = LiveFeed(GrowwFeed(self.groww), self.candle_builder)

        # Execution
        self.broker = Broker(self.groww, settings)
        self.safeguards = SafeguardChecker(settings)
        self.positions = PositionManager(self.groww, settings)

    def _bootstrap_history(self) -> None:
        """Fetch recent historical candles to fill TF buffers on startup."""
        end = datetime.now()
        start = end - timedelta(days=2)
        for symbol in self.symbols:
            df = fetch_historical(
                self.groww, symbol, start, end,
                interval=1,
                exchange=self.settings.groww.exchange,
                segment=self.settings.groww.segment,
            )
            for _, row in df.iterrows():
                candle = row.to_dict()
                self.aggregator.add_candle(symbol, candle)

            # Bootstrap daily data for EMA(200) trend filter
            daily_df = fetch_historical(
                self.groww, symbol,
                end - timedelta(days=400), end,
                interval=1440,
                exchange=self.settings.groww.exchange,
                segment=self.settings.groww.segment,
            )
            if not daily_df.empty:
                self.strategy.daily_df = daily_df

            # Bootstrap 1h history for BB/RSI warmup (~500 hourly bars)
            if "1h" in self.strategy.timeframes:
                hourly_df = fetch_historical(
                    self.groww, symbol,
                    end - timedelta(days=120), end,
                    interval=60,
                    exchange=self.settings.groww.exchange,
                    segment=self.settings.groww.segment,
                )
                for _, row in hourly_df.iterrows():
                    self.aggregator._buffers[symbol]["1h"].append(row.to_dict())

        logger.info("Historical bootstrap complete for %s", self.symbols)

    def _on_candle(self, symbol: str, candle: dict) -> None:
        """Called when CandleBuilder emits a completed 1m candle."""
        self.aggregator.add_candle(symbol, candle)
        self.safeguards.on_candle_close()

        # Build candle dict for strategy
        candles: dict[str, pd.DataFrame] = {}
        candles["1m"] = pd.DataFrame([candle])
        for tf in self.strategy.timeframes:
            if tf != "1m":
                candles[tf] = self.aggregator.get_candles(symbol, tf)

        if not self.strategy.validate_candles(candles):
            return

        signal = self.strategy.on_candle(candles)
        if signal is None:
            return

        # Inject symbol into signal (frozen dataclass)
        signal = dataclasses.replace(signal, symbol=symbol)
        logger.info("Signal: %s", signal)

        # Safeguard check
        allowed, reason = self.safeguards.check(signal, self.settings.trading.capital)
        if not allowed:
            logger.info("Signal blocked: %s", reason)
            return

        # Compute quantity
        price = candle["close"]
        max_value = self.settings.trading.capital * self.settings.safeguards.max_position_pct
        quantity = int(max_value / price)
        if quantity < 1:
            logger.info("Quantity too small for %s at %.2f", symbol, price)
            return

        result = self.broker.place_order(signal, quantity, price)
        logger.info("Order result: %s", result)

        if signal.direction == "BUY":
            self.safeguards.on_position_opened()

            # OCO: single order with TP + SL legs; falls back to separate orders
            if signal.sl_price and signal.tp_price:
                try:
                    oco_result = self.broker.place_oco_order(
                        symbol=signal.symbol,
                        quantity=quantity,
                        tp_price=signal.tp_price,
                        sl_trigger=signal.sl_price,
                    )
                    logger.info("OCO order result: %s", oco_result)
                except Exception:
                    logger.warning("OCO failed, falling back to separate SL+TP", exc_info=True)
                    sl_signal = Signal(direction="SELL", symbol=signal.symbol, strength=1.0)
                    self.broker.place_sl_order(sl_signal, quantity, signal.sl_price)
                    tp_signal = Signal(direction="SELL", symbol=signal.symbol, strength=1.0)
                    self.broker.place_order(tp_signal, quantity, signal.tp_price)

    def run(self) -> None:
        """Start the live trading loop (blocking)."""
        logger.info("Starting live runner (dry_run=%s)", self.settings.dry_run)
        self._bootstrap_history()
        self.positions.sync_positions()
        self.feed.subscribe(self.instruments)
        self.feed.start()
