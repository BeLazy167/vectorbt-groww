"""Live market data feed wrapping GrowwFeed websocket."""

from __future__ import annotations

import logging
import time
from datetime import datetime

from growwapi import GrowwFeed

from data.candle_builder import CandleBuilder

logger = logging.getLogger(__name__)

MAX_RETRIES = 10
BASE_BACKOFF = 1.0


class LiveFeed:
    """Wraps GrowwFeed websocket, routing ticks to a CandleBuilder.

    Args:
        groww_feed: GrowwFeed instance (constructed via GrowwFeed(groww)).
        candle_builder: CandleBuilder to receive LTP ticks.
    """

    def __init__(self, groww_feed: GrowwFeed, candle_builder: CandleBuilder) -> None:
        self._feed = groww_feed
        self._builder = candle_builder
        self._instruments: list[dict] = []
        self._running = False

    def _on_data(self, meta: dict) -> None:
        """Internal callback routed from GrowwFeed.subscribe_ltp."""
        ltp_data = self._feed.get_ltp()
        if not isinstance(ltp_data, dict):
            return
        now = datetime.now()
        for symbol, price in ltp_data.items():
            if isinstance(price, (int, float)) and price > 0:
                self._builder.on_tick(symbol, float(price), now)

    def subscribe(self, instruments: list[dict]) -> None:
        """Subscribe to LTP data for given instruments.

        Args:
            instruments: List of dicts with exchange, segment, exchange_token.
                e.g. [{"exchange": "NSE", "segment": "CASH", "exchange_token": "2885"}]
        """
        self._instruments = instruments
        self._feed.subscribe_ltp(instruments, on_data_received=self._on_data)

    def start(self) -> None:
        """Start the websocket connection (blocking). Reconnects with backoff."""
        self._running = True
        retry_count = 0

        while self._running:
            try:
                logger.info("Starting GrowwFeed consume loop...")
                self._feed.consume()
                break
            except Exception:
                retry_count += 1
                if retry_count > MAX_RETRIES:
                    logger.error("Max retries (%d) exceeded.", MAX_RETRIES)
                    self._running = False
                    raise
                backoff = BASE_BACKOFF * (2 ** (retry_count - 1))
                logger.warning(
                    "Feed disconnected, retry %d/%d in %.1fs",
                    retry_count, MAX_RETRIES, backoff,
                )
                time.sleep(backoff)
                # Re-subscribe after reconnect
                self._feed.subscribe_ltp(
                    self._instruments, on_data_received=self._on_data
                )

    def stop(self) -> None:
        """Stop the feed and flush pending candles."""
        self._running = False
        self._feed.unsubscribe_ltp(self._instruments)
        self._builder.flush_all()
