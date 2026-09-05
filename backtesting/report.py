"""Display and comparison utilities for backtest results."""

from __future__ import annotations

import pandas as pd

from backtesting.engine import BacktestResult


def print_stats(result: BacktestResult) -> None:
    """Print key performance metrics for a backtest result."""
    pf = result.portfolio
    stats = {
        "Strategy": result.strategy_name,
        "Symbol": result.symbol,
        "Total Return": f"{pf.total_return():.2%}",
        "Sharpe Ratio": f"{pf.sharpe_ratio(freq='D'):.4f}",
        "Max Drawdown": f"{pf.max_drawdown():.2%}",
        "Total Trades": pf.orders.records_readable.shape[0],
        "Win Rate": f"{pf.trades.win_rate():.2%}" if pf.trades.count() > 0 else "N/A",
    }
    for k, v in stats.items():
        print(f"  {k:<16} {v}")


def plot_equity(result: BacktestResult) -> None:
    """Plot the equity curve via vectorbt."""
    result.portfolio.plot().show()


def print_trade_log(result: BacktestResult) -> None:
    """Print the full order log as a readable table."""
    print(result.portfolio.orders.records_readable.to_string())


def compare_strategies(results: list[BacktestResult]) -> None:
    """Print a side-by-side comparison table of key metrics.

    Args:
        results: List of BacktestResult from different strategy runs.
    """
    rows = []
    for r in results:
        pf = r.portfolio
        rows.append({
            "Strategy": r.strategy_name,
            "Symbol": r.symbol,
            "Total Return": pf.total_return(),
            "Sharpe Ratio": pf.sharpe_ratio(),
            "Max Drawdown": pf.max_drawdown(),
            "Total Trades": pf.orders.records_readable.shape[0],
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
