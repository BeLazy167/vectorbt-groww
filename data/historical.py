"""Fetch historical candle data from Groww API with automatic pagination and SQLite cache."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
from growwapi import GrowwAPI

from data.cache import get_cached_candles, store_candles

logger = logging.getLogger(__name__)

_INTERVAL_TO_STR = {
    1: "1minute", 2: "2minute", 3: "3minute", 5: "5minute",
    10: "10minute", 15: "15minute", 30: "30minute",
    60: "1hour", 240: "4hour", 1440: "1day", 10080: "1week",
}

_MAX_DAYS = {
    1: 30, 5: 30, 10: 30, 15: 30, 30: 30,
    60: 180, 240: 365, 1440: 1080, 10080: 3650,
}


def fetch_historical(
    groww: GrowwAPI,
    symbol: str,
    start: datetime,
    end: datetime,
    interval: int = 1,
    exchange: str = "NSE",
    segment: str = "CASH",
) -> pd.DataFrame:
    """Fetch historical OHLCV candles with cache-first strategy.

    Uses the backtesting API (get_historical_candles) with groww_symbol format.
    """
    now = datetime.now()
    if end > now:
        end = now

    cached = get_cached_candles(symbol, exchange, segment, interval, start, end)
    if cached is not None and not cached.empty:
        logger.info("Cache hit: %s %s %dm (%d rows)", symbol, exchange, interval, len(cached))
        return cached

    logger.info("Cache miss: fetching %s %s %dm from API", symbol, exchange, interval)
    groww_symbol = f"{exchange}-{symbol}"
    candle_interval = _INTERVAL_TO_STR.get(interval, f"{interval}minute")
    max_days = _MAX_DAYS.get(interval, 30)
    chunks: list[pd.DataFrame] = []
    chunk_start = start

    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=max_days), end)
        start_str = chunk_start.strftime("%Y-%m-%d %H:%M:%S")
        end_str = chunk_end.strftime("%Y-%m-%d %H:%M:%S")
        logger.info("  Fetching %s -> %s", start_str, end_str)
        resp = groww.get_historical_candles(
            exchange=exchange,
            segment=segment,
            groww_symbol=groww_symbol,
            start_time=start_str,
            end_time=end_str,
            candle_interval=candle_interval,
        )
        rows = resp.get("candles", []) if isinstance(resp, dict) else resp
        if rows:
            # Response: [timestamp, open, high, low, close, volume, open_interest]
            cols = ["timestamp", "open", "high", "low", "close", "volume", "oi"]
            df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
            if "oi" in df.columns:
                df = df.drop(columns=["oi"])
            chunks.append(df)
        chunk_start = chunk_end

    if not chunks:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    result = pd.concat(chunks, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], unit="s")
    result = result.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    store_candles(symbol, exchange, segment, interval, result)
    logger.info("Cached %d candles for %s", len(result), symbol)

    return result
