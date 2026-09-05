"""SQLite cache for historical candles and instruments data."""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "cache.db"

INSTRUMENTS_TTL_HOURS = 24


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _ensure_tables(conn)
    return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            segment TEXT NOT NULL,
            interval_min INTEGER NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, exchange, segment, interval_min, timestamp)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS instruments (
            data BLOB NOT NULL,
            updated_at REAL NOT NULL
        )
    """)
    conn.commit()


# ── Candles ──────────────────────────────────────────────


def get_cached_candles(
    symbol: str,
    exchange: str,
    segment: str,
    interval: int,
    start: datetime,
    end: datetime,
) -> pd.DataFrame | None:
    """Return cached candles for the full range, or None if incomplete."""
    conn = _get_conn()
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())

    rows = conn.execute(
        """SELECT timestamp, open, high, low, close, volume
           FROM candles
           WHERE symbol=? AND exchange=? AND segment=? AND interval_min=?
             AND timestamp >= ? AND timestamp <= ?
           ORDER BY timestamp""",
        (symbol, exchange, segment, interval, start_ts, end_ts),
    ).fetchall()
    conn.close()

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def store_candles(
    symbol: str,
    exchange: str,
    segment: str,
    interval: int,
    df: pd.DataFrame,
) -> None:
    """Upsert candle rows into cache."""
    if df.empty:
        return
    conn = _get_conn()
    rows = []
    for _, r in df.iterrows():
        ts = int(r["timestamp"].timestamp()) if isinstance(r["timestamp"], pd.Timestamp) else int(r["timestamp"])
        rows.append((symbol, exchange, segment, interval, ts, r["open"], r["high"], r["low"], r["close"], r["volume"]))

    conn.executemany(
        """INSERT OR REPLACE INTO candles
           (symbol, exchange, segment, interval_min, timestamp, open, high, low, close, volume)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    conn.close()


# ── Instruments ──────────────────────────────────────────


def get_cached_instruments() -> pd.DataFrame | None:
    """Return cached instruments if fresh (within TTL), else None."""
    conn = _get_conn()
    row = conn.execute("SELECT data, updated_at FROM instruments ORDER BY rowid DESC LIMIT 1").fetchone()
    conn.close()

    if row is None:
        return None

    age_hours = (time.time() - row[1]) / 3600
    if age_hours > INSTRUMENTS_TTL_HOURS:
        return None

    return pd.read_json(row[0], orient="split")


def store_instruments(df: pd.DataFrame) -> None:
    """Store instruments DataFrame (replaces previous)."""
    conn = _get_conn()
    conn.execute("DELETE FROM instruments")
    blob = df.to_json(orient="split")
    conn.execute("INSERT INTO instruments (data, updated_at) VALUES (?, ?)", (blob, time.time()))
    conn.commit()
    conn.close()
