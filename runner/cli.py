"""CLI interface for backtest, live, and report commands."""

from __future__ import annotations

import logging
from datetime import datetime

import click

from config.settings import Settings


@click.group()
@click.option("--log-level", default="INFO", help="Logging level.")
@click.pass_context
def cli(ctx: click.Context, log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    ctx.ensure_object(dict)
    ctx.obj["settings"] = Settings(log_level=log_level.upper())


@cli.command()
@click.option("--strategy", required=True, help="Strategy name.")
@click.option("--symbol", required=True, help="Trading symbol (e.g. RELIANCE).")
@click.option("--start", required=True, help="Start date YYYY-MM-DD.")
@click.option("--end", required=True, help="End date YYYY-MM-DD.")
@click.option("--capital", default=100_000.0, help="Initial capital.")
@click.option("--interval", default=1, help="Candle interval in minutes (1, 5, 15).")
@click.pass_context
def backtest(
    ctx: click.Context,
    strategy: str,
    symbol: str,
    start: str,
    end: str,
    capital: float,
    interval: int,
) -> None:
    """Run a backtest for a strategy over a date range."""
    # Import strategies to trigger registry decorators
    import strategies.bb_mean_reversion  # noqa: F401
    import strategies.ma_crossover  # noqa: F401
    import strategies.multi_confluence  # noqa: F401
    import strategies.orb_rsi  # noqa: F401
    from backtesting.engine import BacktestEngine
    from backtesting.report import plot_equity, print_stats, print_trade_log

    settings: Settings = ctx.obj["settings"]
    settings.trading.capital = capital

    engine = BacktestEngine(settings)
    result = engine.run(
        strategy_name=strategy,
        symbol=symbol,
        start=datetime.strptime(start, "%Y-%m-%d"),
        end=datetime.strptime(end, "%Y-%m-%d"),
        interval=interval,
    )

    print_stats(result)
    print_trade_log(result)
    plot_equity(result)


@cli.command()
@click.option("--strategy", required=True, help="Strategy name.")
@click.option("--symbols", required=True, help="Comma-separated trading symbols.")
@click.option("--capital", default=100_000.0, help="Trading capital.")
@click.option("--dry-run/--no-dry-run", default=True, help="Paper trading mode.")
@click.pass_context
def live(
    ctx: click.Context,
    strategy: str,
    symbols: str,
    capital: float,
    dry_run: bool,
) -> None:
    """Start live trading with a strategy."""
    import strategies.bb_mean_reversion  # noqa: F401
    import strategies.ma_crossover  # noqa: F401
    import strategies.multi_confluence  # noqa: F401
    import strategies.orb_rsi  # noqa: F401
    from runner.live import LiveRunner

    settings: Settings = ctx.obj["settings"]
    settings.trading.capital = capital
    settings.dry_run = dry_run

    symbol_list = [s.strip() for s in symbols.split(",")]

    # Build instrument list — user must provide exchange_tokens via env or config
    # For now, use symbol names as placeholders
    instruments = [
        {"exchange": "NSE", "segment": "CASH", "exchange_token": sym}
        for sym in symbol_list
    ]

    runner = LiveRunner(settings, strategy, symbol_list, instruments)
    runner.run()


from runner.options_cli import options as options_group

cli.add_command(options_group)


@cli.command()
@click.option("--strategy", required=True, help="Strategy name.")
@click.option("--symbol", required=True, help="Trading symbol.")
@click.option("--start", required=True, help="Start date YYYY-MM-DD.")
@click.option("--end", required=True, help="End date YYYY-MM-DD.")
@click.pass_context
def report(
    ctx: click.Context,
    strategy: str,
    symbol: str,
    start: str,
    end: str,
) -> None:
    """Generate a detailed report for a backtest run."""
    import strategies.bb_mean_reversion  # noqa: F401
    import strategies.ma_crossover  # noqa: F401
    import strategies.multi_confluence  # noqa: F401
    import strategies.orb_rsi  # noqa: F401
    from backtesting.engine import BacktestEngine
    from backtesting.report import compare_strategies, print_stats

    settings: Settings = ctx.obj["settings"]
    engine = BacktestEngine(settings)
    result = engine.run(
        strategy_name=strategy,
        symbol=symbol,
        start=datetime.strptime(start, "%Y-%m-%d"),
        end=datetime.strptime(end, "%Y-%m-%d"),
    )

    print_stats(result)
