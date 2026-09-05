"""CLI subgroup for options commands."""

from __future__ import annotations

import logging
from datetime import datetime

import click

from config.settings import Settings

logger = logging.getLogger(__name__)


@click.group()
@click.pass_context
def options(ctx: click.Context) -> None:
    """Options / volatility trading commands."""
    pass


@options.command()
@click.option("--strategy", required=True, help="Options strategy name.")
@click.option("--underlying", required=True, help="Underlying symbol (e.g. NIFTY).")
@click.option("--start", required=True, help="Start date YYYY-MM-DD.")
@click.option("--end", required=True, help="End date YYYY-MM-DD.")
@click.option("--capital", default=100_000.0, help="Initial capital.")
@click.pass_context
def backtest(
    ctx: click.Context,
    strategy: str,
    underlying: str,
    start: str,
    end: str,
    capital: float,
) -> None:
    """Run an options backtest over expiry cycles."""
    from backtesting.options_engine import OptionsBacktestEngine
    from backtesting.options_report import plot_options_equity, print_options_stats
    from strategies.vrp_butterfly import VRPButterflyStrategy  # noqa: F401
    from strategies.vrp_straddle import VRPStraddleStrategy  # noqa: F401
    from strategies.registry import get_options

    settings: Settings = ctx.obj["settings"]
    settings.trading.capital = capital

    strategy_cls = get_options(strategy)
    strat = strategy_cls(settings.options, settings.vrp)

    engine = OptionsBacktestEngine(settings, settings.options, settings.vrp)
    result = engine.run(
        strategy=strat,
        underlying=underlying,
        start=datetime.strptime(start, "%Y-%m-%d"),
        end=datetime.strptime(end, "%Y-%m-%d"),
    )

    print_options_stats(result)
    plot_options_equity(result)


@options.command()
@click.option("--underlying", required=True, help="Underlying symbol.")
@click.option("--start", required=True, help="Start date YYYY-MM-DD.")
@click.option("--end", required=True, help="End date YYYY-MM-DD.")
@click.pass_context
def research(ctx: click.Context, underlying: str, start: str, end: str) -> None:
    """Generate vol research analytics (vol cone, IVRV, IV percentile)."""
    from growwapi import GrowwAPI

    from data.historical import fetch_historical
    from backtesting.options_report import plot_vol_cone
    from data.iv_engine import realized_vol

    settings: Settings = ctx.obj["settings"]
    token = settings.groww.get_token()
    groww = GrowwAPI(token)

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    df = fetch_historical(groww, underlying, start_dt, end_dt, interval=1440)
    rv = realized_vol(df["close"], window=settings.options.rv_window)

    click.echo(f"Realized Vol (latest): {rv.iloc[-1]:.4f}")
    click.echo(f"RV mean: {rv.mean():.4f}, RV std: {rv.std():.4f}")
    plot_vol_cone(rv, rv)


@options.command()
@click.option("--underlying", required=True, help="Underlying symbol.")
@click.option("--model-type", default="regression", help="regression or classification.")
@click.option("--output", required=True, help="Path to save model pkl.")
@click.pass_context
def train(
    ctx: click.Context, underlying: str, model_type: str, output: str
) -> None:
    """Train a VRP prediction model."""
    click.echo(f"Training {model_type} model for {underlying}...")
    click.echo(f"Model will be saved to {output}")
    click.echo("NOTE: Requires pre-computed feature matrix. See models/pipeline.py")


@options.command()
@click.option("--min-volume", default=1000, help="Minimum option volume.")
@click.option("--min-oi", default=5000, help="Minimum open interest.")
@click.pass_context
def screener(ctx: click.Context, min_volume: int, min_oi: int) -> None:
    """Screen F&O universe by liquidity."""
    from growwapi import GrowwAPI

    from data.universe import screen_universe

    settings: Settings = ctx.obj["settings"]
    token = settings.groww.get_token()
    groww = GrowwAPI(token)

    underlyings = ["NIFTY", "BANKNIFTY", "FINNIFTY", "RELIANCE", "TCS", "INFY",
                    "HDFCBANK", "ICICIBANK", "SBIN", "TATAMOTORS"]

    qualifying = screen_universe(groww, underlyings, min_volume, min_oi)
    click.echo(f"Qualifying instruments ({len(qualifying)}):")
    for sym in qualifying:
        click.echo(f"  {sym}")


@options.command(name="live")
@click.option("--underlyings", required=True, help="Comma-separated underlyings.")
@click.option("--strategy", required=True, help="Options strategy name.")
@click.option("--dry-run/--no-dry-run", default=True, help="Paper trading mode.")
@click.pass_context
def live_cmd(
    ctx: click.Context, underlyings: str, strategy: str, dry_run: bool
) -> None:
    """Start live VRP trading."""
    from runner.live_vrp import LiveVRPRunner
    from strategies.vrp_butterfly import VRPButterflyStrategy  # noqa: F401
    from strategies.vrp_straddle import VRPStraddleStrategy  # noqa: F401

    settings: Settings = ctx.obj["settings"]
    settings.dry_run = dry_run

    underlying_list = [s.strip() for s in underlyings.split(",")]
    runner = LiveVRPRunner(settings, strategy, underlying_list, dry_run)
    runner.run()
