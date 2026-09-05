"""F&O historical candle fetcher with 7-day pagination."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from growwapi import GrowwAPI

MAX_DAYS_PER_REQUEST = 7


def fetch_fno_candles(
    groww: GrowwAPI,
    trading_symbol: str,
    start: datetime,
    end: datetime,
    interval: int = 1,
    exchange: str = "NSE",
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    chunk_start = start

    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=MAX_DAYS_PER_REQUEST), end)
        start_str = chunk_start.strftime("%Y-%m-%d %H:%M:%S")
        end_str = chunk_end.strftime("%Y-%m-%d %H:%M:%S")
        resp = groww.get_historical_candle_data(
            trading_symbol=trading_symbol,
            exchange=exchange,
            segment="FNO",
            start_time=start_str,
            end_time=end_str,
            interval_in_minutes=interval,
        )
        rows = resp.get("candles", []) if isinstance(resp, dict) else resp
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        if not df.empty:
            chunks.append(df)
        chunk_start = chunk_end

    if not chunks:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    result = pd.concat(chunks, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    return result.sort_values("timestamp").reset_index(drop=True)
