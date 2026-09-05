"""Live VRP trading runner with polling loop."""

from __future__ import annotations

import logging
import time
from datetime import datetime

from growwapi import GrowwAPI

from config.settings import Settings
from config.options_config import VRPConfig
from data.chain import OptionChain, fetch_expiries, fetch_option_chain
from execution.delta_hedger import DeltaHedger
from execution.options_broker import OptionsBroker
from execution.options_safeguards import OptionsSafeguardChecker
from models.predict import predict_vrp
from signals.composite import compute_snapshot, is_vrp_entry
from strategies.registry import get_options

logger = logging.getLogger(__name__)

# Default lot sizes for major NSE F&O instruments
LOT_SIZES: dict[str, int] = {
    "NIFTY": 25,
    "BANKNIFTY": 15,
    "FINNIFTY": 25,
    "MIDCPNIFTY": 50,
}


class LiveVRPRunner:
    """Polling-based live options trading loop.

    Periodically fetches option chains, computes vol signals,
    and executes VRP strategy entries/exits.
    """

    def __init__(
        self,
        settings: Settings,
        strategy_name: str,
        underlyings: list[str],
        dry_run: bool = True,
        poll_interval: int = 300,
        model_path: str | None = None,
    ) -> None:
        self._settings = settings
        self._strategy_name = strategy_name
        self._underlyings = underlyings
        self._dry_run = dry_run
        self._poll_interval = poll_interval
        self._model_path = model_path

        # Auth
        token = settings.groww.get_token()
        self._groww = GrowwAPI(token)

        # Components
        strategy_cls = get_options(strategy_name)
        self._strategy = strategy_cls(settings.options, settings.vrp)
        self._broker = OptionsBroker(self._groww, settings)
        self._safeguards = OptionsSafeguardChecker(
            settings.vrp, settings.safeguards
        )
        self._hedger = DeltaHedger(self._groww, settings, settings.vrp)

        # State
        self._open_positions: dict[str, dict] = {}  # underlying -> position info
        self._iv_history: dict[str, list[float]] = {}  # underlying -> recent IVs

    def run(self) -> None:
        """Start the polling loop (blocking)."""
        logger.info(
            "Starting live VRP runner: underlyings=%s, strategy=%s, dry_run=%s",
            self._underlyings, self._strategy_name, self._dry_run,
        )

        while True:
            for underlying in self._underlyings:
                self._process_underlying(underlying)
            time.sleep(self._poll_interval)

    def _process_underlying(self, underlying: str) -> None:
        try:
            self._process_underlying_inner(underlying)
        except Exception:
            logger.exception("Error processing %s, skipping cycle", underlying)

    def _process_underlying_inner(self, underlying: str) -> None:
        now = datetime.now()
        lot_size = LOT_SIZES.get(underlying, 25)

        # Fetch nearest expiry
        expiries = fetch_expiries(self._groww, underlying, now.year)
        future_expiries = [
            e for e in expiries
            if datetime.strptime(e, "%Y-%m-%d") > now
        ]
        if not future_expiries:
            logger.warning("No future expiries found for %s", underlying)
            return
        nearest_expiry = future_expiries[0]

        # Fetch chain
        chain = fetch_option_chain(self._groww, underlying, nearest_expiry)

        # Check existing position for exit/hedge
        if underlying in self._open_positions:
            self._manage_position(underlying, chain, lot_size)
            return

        # Compute vol snapshot for entry
        from data.historical import fetch_historical
        from datetime import timedelta

        start = now - timedelta(days=365)
        df = fetch_historical(
            self._groww, underlying, start, now, interval=1440
        )
        if df.empty:
            logger.warning("No historical data for %s", underlying)
            return

        iv_hist = self._iv_history.get(underlying)
        import pandas as pd
        iv_hist_series = pd.Series(iv_hist) if iv_hist else pd.Series(dtype=float)

        snapshot = compute_snapshot(
            chain, df["close"], iv_hist_series, rv_window=self._settings.options.rv_window
        )

        # Track IV history
        self._iv_history.setdefault(underlying, []).append(snapshot.atm_iv)

        # Model prediction (optional)
        model_score = None
        if self._model_path:
            model_score = predict_vrp(snapshot, self._model_path, self._settings.model)

        # Generate signal
        signal = self._strategy.generate_signal(
            chain, df["close"], snapshot, model_score
        )
        if signal is None:
            logger.debug("No signal for %s", underlying)
            return

        # Safeguard check
        allowed, reason = self._safeguards.check(signal, chain, self._settings.trading.capital, lot_size)
        if not allowed:
            logger.info("Signal blocked for %s: %s", underlying, reason)
            return

        # Place order
        results = self._broker.place_legs(signal, lot_size)
        logger.info("Orders placed for %s: %s", underlying, results)

        if any(r.get("status") == "FAILED" for r in results):
            logger.error("Order placement failed for %s, not tracking position", underlying)
            return

        self._open_positions[underlying] = {
            "signal": signal,
            "entry_date": now,
            "entry_premium": chain.straddle_premium(),
        }

    def _manage_position(
        self, underlying: str, chain: OptionChain, lot_size: int
    ) -> None:
        pos = self._open_positions[underlying]
        days_in_trade = (datetime.now() - pos["entry_date"]).days
        entry_premium = pos["entry_premium"]
        current_premium = chain.straddle_premium()
        current_pnl = entry_premium - current_premium  # short straddle PnL

        # Check exit
        if self._strategy.exit_condition(
            chain, pos["signal"], days_in_trade, current_pnl, self._settings.vrp
        ):
            logger.info(
                "Exiting %s: days=%d, pnl=%.2f",
                underlying, days_in_trade, current_pnl,
            )
            # Place closing orders (reverse legs)
            from signals.base import OptionsSignal, OptionLeg, StrategyAction
            close_legs = tuple(
                OptionLeg(
                    trading_symbol=leg.trading_symbol,
                    strike=leg.strike,
                    option_type=leg.option_type,
                    action=StrategyAction.BUY if leg.action == StrategyAction.SELL else StrategyAction.SELL,
                    lots=leg.lots,
                )
                for leg in pos["signal"].legs
            )
            close_signal = OptionsSignal(
                underlying=pos["signal"].underlying,
                expiry=pos["signal"].expiry,
                legs=close_legs,
                strategy_name=pos["signal"].strategy_name,
                signal_strength=0.0,
                entry_iv=0.0,
                entry_ivrv=0.0,
            )
            close_results = self._broker.place_legs(close_signal, lot_size)
            if any(r.get("status") == "FAILED" for r in close_results):
                logger.error("Close order failed for %s, position still tracked. Results: %s", underlying, close_results)
                return
            self._safeguards.on_trade_closed(current_pnl)
            del self._open_positions[underlying]
            return

        # Delta hedge check
        positions = [
            {
                "strike": leg.strike,
                "option_type": leg.option_type.value,
                "action": leg.action.value,
                "lots": leg.lots,
            }
            for leg in pos["signal"].legs
        ]
        self._hedger.check_and_hedge(chain, positions, lot_size)
