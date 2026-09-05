"""Options backtest reporting and visualization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtesting.options_engine import OptionsBacktestResult


def print_options_stats(result: OptionsBacktestResult) -> None:
    """Print summary statistics table."""
    n = len(result.trades)
    print("\n" + "=" * 50)
    print("OPTIONS BACKTEST SUMMARY")
    print("=" * 50)
    print(f"  Trades:        {n}")
    print(f"  Total PnL:     {result.total_pnl:>12,.2f}")
    print(f"  Win Rate:      {result.win_rate:>11.1%}")
    print(f"  Avg PnL:       {result.avg_pnl:>12,.2f}")
    print(f"  Max Drawdown:  {result.max_drawdown:>12,.2f}")
    print(f"  Sharpe:        {result.sharpe:>12.3f}")
    print("=" * 50 + "\n")


def plot_options_equity(result: OptionsBacktestResult) -> None:
    """Plot cumulative PnL equity curve."""
    if not result.trades:
        print("No trades to plot.")
        return

    pnls = [t["pnl"] for t in result.trades]
    cum_pnl = np.cumsum(pnls)
    labels = [t.get("expiry", str(i)) for i, t in enumerate(result.trades)]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(cum_pnl)), cum_pnl, marker="o", linewidth=1.5, markersize=4)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.fill_between(range(len(cum_pnl)), cum_pnl, alpha=0.15)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative PnL")
    ax.set_title("Options Strategy Equity Curve")

    if len(labels) <= 30:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_vol_cone(iv_series: pd.Series, rv_series: pd.Series) -> None:
    """Plot IV vs RV time series comparison."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(iv_series.index, iv_series.values, label="IV", linewidth=1.2)
    ax.plot(rv_series.index, rv_series.values, label="RV", linewidth=1.2)
    ax.fill_between(
        iv_series.index,
        iv_series.values,
        rv_series.reindex(iv_series.index).values,
        alpha=0.15,
        label="IV-RV Spread",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.set_title("Implied vs Realized Volatility")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
