"""Batch backtest ORB+RSI across F&O universe."""

from __future__ import annotations

import csv
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

import strategies.orb_rsi  # noqa: F401
from backtesting.engine import BacktestEngine
from config.settings import Settings

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STRATEGY = "orb_rsi"
INTERVAL = 5  # 5m candles
CAPITAL = 100_000.0

# 3 months back from today
END = datetime.now()
START = END - timedelta(days=90)


def load_fno_symbols() -> list[str]:
    """Extract unique F&O underlying symbols from Groww instruments CSV."""
    import importlib.resources as pkg_resources
    csv_path = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "growwapi" / "instruments.csv"
    symbols = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["segment"] == "FNO" and row["instrument_type"] == "FUT":
                sym = row["underlying_symbol"]
                if sym and "NSETEST" not in sym:
                    symbols.add(sym)
    return sorted(symbols)


def load_symbols_from_file(path: str) -> list[str]:
    """Load symbols from a text file, one per line."""
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def main() -> None:
    # Accept optional symbol file as CLI arg
    if len(sys.argv) > 1:
        symbols = load_symbols_from_file(sys.argv[1])
    else:
        symbols = load_fno_symbols()
    print(f"Backtesting {len(symbols)} F&O symbols from {START:%Y-%m-%d} to {END:%Y-%m-%d}")
    print(f"Strategy: {STRATEGY} | Interval: {INTERVAL}m | Capital: {CAPITAL:,.0f}\n")

    settings = Settings()
    settings.trading.capital = CAPITAL
    engine = BacktestEngine(settings)

    rows = []
    errors = []

    for i, symbol in enumerate(symbols, 1):
        try:
            result = engine.run(STRATEGY, symbol, START, END, interval=INTERVAL)
            pf = result.portfolio
            trades = pf.trades.count()
            wr = pf.trades.win_rate() if trades > 0 else 0.0
            total_ret = pf.total_return()
            pnl = pf.total_profit()
            sharpe = pf.sharpe_ratio(freq="D")
            max_dd = pf.max_drawdown()

            rows.append({
                "symbol": symbol,
                "trades": int(trades),
                "win_rate": round(wr, 4),
                "total_return": round(total_ret, 4),
                "pnl": round(float(pnl), 2),
                "sharpe": round(float(sharpe), 4) if pd.notna(sharpe) else 0.0,
                "max_dd": round(float(max_dd), 4),
            })
            status = f"✓ {trades}T {wr:.0%}WR {pnl:+,.0f}"
        except Exception as e:
            errors.append((symbol, str(e)))
            status = f"✗ {e}"

        print(f"[{i:3d}/{len(symbols)}] {symbol:<20} {status}")

    # Summary
    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo results.")
        return

    print(f"\n{'='*60}")
    print(f"BATCH RESULTS: {len(df)} symbols | {len(errors)} errors")
    print(f"{'='*60}")

    profitable = df[df["pnl"] > 0]
    print(f"Profitable symbols:  {len(profitable)}/{len(df)} ({len(profitable)/len(df):.0%})")
    print(f"Total PnL:           {df['pnl'].sum():+,.0f}")
    print(f"Avg PnL/symbol:      {df['pnl'].mean():+,.0f}")
    print(f"Avg Win Rate:        {df[df['trades']>0]['win_rate'].mean():.1%}")
    print(f"Avg Trades/symbol:   {df['trades'].mean():.1f}")
    print(f"Total Trades:        {df['trades'].sum()}")
    print(f"Median Sharpe:       {df['sharpe'].median():.4f}")

    # Top/bottom
    print(f"\n--- Top 10 by PnL ---")
    top = df.nlargest(10, "pnl")
    print(top[["symbol", "trades", "win_rate", "pnl", "sharpe"]].to_string(index=False))

    print(f"\n--- Bottom 10 by PnL ---")
    bot = df.nsmallest(10, "pnl")
    print(bot[["symbol", "trades", "win_rate", "pnl", "sharpe"]].to_string(index=False))

    # Save full results
    out_path = Path("backtest_fno_results.csv")
    df.sort_values("pnl", ascending=False).to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
