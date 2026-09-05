"""Expiry-cycle options backtest engine."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from growwapi import GrowwAPI

from config.settings import Settings
from data.historical import fetch_historical
from backtesting.options_pnl import trade_pnl
from config.options_config import OptionsConfig, VRPConfig
from data.chain import OptionChain, fetch_expiries, fetch_option_chain
from data.fno_historical import fetch_fno_candles
from data.iv_engine import implied_vol, realized_vol
from signals.composite import VolSignalSnapshot, compute_snapshot
from strategies.options_base import OptionsStrategy

logger = logging.getLogger(__name__)

LOT_SIZE_MAP: dict[str, int] = {
    "NIFTY": 25,
    "BANKNIFTY": 15,
    "FINNIFTY": 25,
    "MIDCPNIFTY": 50,
}
DEFAULT_LOT_SIZE = 25


@dataclass
class OptionsBacktestResult:
    trades: list[dict] = field(default_factory=list)
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0


class OptionsBacktestEngine:
    """Runs options backtests across expiry cycles.

    Args:
        settings: Global pipeline settings for Groww auth.
        options_config: Options-specific config.
        vrp_config: VRP strategy thresholds.
    """

    def __init__(
        self,
        settings: Settings,
        options_config: OptionsConfig,
        vrp_config: VRPConfig,
    ) -> None:
        self.settings = settings
        self.options_config = options_config
        self.vrp_config = vrp_config

    def run(
        self,
        strategy: OptionsStrategy,
        underlying: str,
        start: datetime,
        end: datetime,
    ) -> OptionsBacktestResult:
        """Run backtest across expiry cycles.

        Args:
            strategy: Options strategy instance.
            underlying: Underlying symbol (e.g. "NIFTY").
            start: Backtest start date.
            end: Backtest end date.

        Returns:
            OptionsBacktestResult with trade-level and aggregate metrics.
        """
        token = self.settings.groww.get_token()
        groww = GrowwAPI(token)

        # Extend lookback for RV calculation (need rv_window + buffer trading days)
        lookback_start = start - timedelta(days=int(self.options_config.rv_window * 1.8))
        underlying_df = fetch_historical(
            groww, underlying, lookback_start, end,
            interval=1440,
            exchange=self.settings.groww.exchange,
        )
        underlying_df = underlying_df.set_index("timestamp")
        underlying_prices = underlying_df["close"]

        expiries = self._collect_expiries(groww, underlying, start, end)
        lot_size = LOT_SIZE_MAP.get(underlying, DEFAULT_LOT_SIZE)

        rv_series = realized_vol(underlying_prices, self.options_config.rv_window)

        trades: list[dict] = []

        for expiry_str in expiries:
            expiry_date = pd.Timestamp(expiry_str)
            if expiry_date < pd.Timestamp(start) or expiry_date > pd.Timestamp(end):
                continue

            entry_date = expiry_date - timedelta(days=self.options_config.days_before_expiry_entry)
            exit_deadline = expiry_date - timedelta(days=self.options_config.days_before_expiry_exit)

            if entry_date < pd.Timestamp(start):
                continue

            trade = self._run_expiry_cycle(
                groww, strategy, underlying, underlying_prices,
                rv_series, expiry_str, entry_date, exit_deadline, lot_size,
            )
            if trade is not None:
                trades.append(trade)

        return self._aggregate(trades)

    def _collect_expiries(
        self, groww: GrowwAPI, underlying: str, start: datetime, end: datetime
    ) -> list[str]:
        expiries: list[str] = []
        for year in range(start.year, end.year + 1):
            start_month = start.month if year == start.year else 1
            end_month = end.month if year == end.year else 12
            for month in range(start_month, end_month + 1):
                try:
                    batch = fetch_expiries(groww, underlying, year, month)
                    expiries.extend(batch)
                except Exception:
                    logger.warning("No expiries for %s %d-%02d", underlying, year, month)
        return sorted(set(expiries))

    def _run_expiry_cycle(
        self,
        groww: GrowwAPI,
        strategy: OptionsStrategy,
        underlying: str,
        underlying_prices: pd.Series,
        rv_series: pd.Series,
        expiry_str: str,
        entry_date: pd.Timestamp,
        exit_deadline: pd.Timestamp,
        lot_size: int,
    ) -> dict | None:
        """Execute a single expiry cycle: entry check, walk-forward, exit."""
        try:
            chain = fetch_option_chain(groww, underlying, expiry_str)
        except Exception:
            logger.warning("Failed to fetch chain for %s expiry %s", underlying, expiry_str)
            return None

        iv_history = pd.Series(dtype=float)
        try:
            snapshot = compute_snapshot(chain, underlying_prices, iv_history)
        except Exception:
            logger.warning("Failed to compute snapshot for %s expiry %s", underlying, expiry_str)
            return None

        signal = strategy.generate_signal(chain, underlying_prices, snapshot)
        if signal is None:
            return None

        entry_prices: dict[str, float] = {}
        for leg in signal.legs:
            contract = chain.get_contract(leg.strike, leg.option_type)
            if contract is None:
                return None
            entry_prices[leg.trading_symbol] = contract.ltp

        exit_prices: dict[str, float] = {}
        exit_date = entry_date
        days_in_trade = 0
        incomplete = False

        current_date = entry_date + timedelta(days=1)
        while current_date <= exit_deadline:
            days_in_trade += 1

            try:
                current_chain = fetch_option_chain(groww, underlying, expiry_str)
            except Exception:
                current_date += timedelta(days=1)
                continue

            current_exit_prices: dict[str, float] = {}
            all_found = True
            for leg in signal.legs:
                contract = current_chain.get_contract(leg.strike, leg.option_type)
                if contract is None:
                    all_found = False
                    break
                current_exit_prices[leg.trading_symbol] = contract.ltp

            if not all_found:
                current_date += timedelta(days=1)
                continue

            current_pnl = trade_pnl(signal.legs, entry_prices, current_exit_prices, lot_size)

            if strategy.exit_condition(current_chain, signal, days_in_trade, current_pnl, self.vrp_config):
                exit_prices = current_exit_prices
                exit_date = current_date
                break

            current_date += timedelta(days=1)
        else:
            # Reached exit deadline without early exit
            exit_date = exit_deadline
            incomplete = False
            try:
                final_chain = fetch_option_chain(groww, underlying, expiry_str)
                for leg in signal.legs:
                    contract = final_chain.get_contract(leg.strike, leg.option_type)
                    if contract is None:
                        incomplete = True
                        logger.warning("Missing exit contract for %s in %s expiry %s", leg.trading_symbol, underlying, expiry_str)
                    exit_prices[leg.trading_symbol] = contract.ltp if contract else 0.0
            except Exception:
                incomplete = True
                logger.warning("Failed to fetch final chain for %s expiry %s, trade marked incomplete", underlying, expiry_str)
                for leg in signal.legs:
                    exit_prices.setdefault(leg.trading_symbol, 0.0)

        pnl = trade_pnl(signal.legs, entry_prices, exit_prices, lot_size)

        return {
            "expiry": expiry_str,
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()) if isinstance(exit_date, pd.Timestamp) else str(exit_date),
            "entry_prices": entry_prices,
            "exit_prices": exit_prices,
            "pnl": pnl,
            "incomplete": incomplete,
            "vol_snapshot_dict": {
                "atm_iv": snapshot.atm_iv,
                "rv_30d": snapshot.rv_30d,
                "ivrv_ratio": snapshot.ivrv_ratio,
                "iv_percentile_rank": snapshot.iv_percentile_rank,
                "days_to_expiry": snapshot.days_to_expiry,
            },
        }

    @staticmethod
    def _aggregate(trades: list[dict]) -> OptionsBacktestResult:
        """Compute aggregate metrics from trade list."""
        result = OptionsBacktestResult(trades=trades)
        if not trades:
            return result

        pnls = [t["pnl"] for t in trades]
        result.total_pnl = sum(pnls)
        result.win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
        result.avg_pnl = result.total_pnl / len(pnls)

        # Max drawdown from cumulative PnL
        cum_pnl = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum_pnl)
        drawdowns = peak - cum_pnl
        result.max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Sharpe: annualize based on actual trade frequency
        pnl_arr = np.array(pnls)
        std = float(np.std(pnl_arr, ddof=1)) if len(pnl_arr) > 1 else 0.0
        if len(trades) >= 2:
            first = pd.Timestamp(trades[0]["expiry"])
            last = pd.Timestamp(trades[-1]["expiry"])
            years = max((last - first).days / 365.0, 0.1)
            trades_per_year = len(pnls) / years
        else:
            trades_per_year = 12
        if std > 0:
            result.sharpe = (float(np.mean(pnl_arr)) / std) * math.sqrt(trades_per_year)

        return result
