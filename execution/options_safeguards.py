"""Greeks-based safeguard checks for options trading."""

from __future__ import annotations

import logging
from datetime import datetime, time

from zoneinfo import ZoneInfo

from config.settings import SafeguardConfig
from config.options_config import VRPConfig
from data.chain import OptionChain, OptionContract
from signals.base import OptionsSignal, OptionType

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")


class OptionsSafeguardChecker:
    """Validates greeks exposure and trading conditions for options."""

    def __init__(
        self, vrp_config: VRPConfig, safeguard_config: SafeguardConfig
    ) -> None:
        self._vrp = vrp_config
        self._sg = safeguard_config
        self._daily_pnl: float = 0.0
        self._portfolio_delta: float = 0.0
        self._portfolio_vega: float = 0.0
        self._portfolio_gamma: float = 0.0

    def check(
        self,
        signal: OptionsSignal,
        chain: OptionChain,
        capital: float,
        lot_size: int,
    ) -> tuple[bool, str]:
        """Run all safeguard checks for an options signal.

        Args:
            signal: Proposed options trade.
            chain: Current option chain for greeks lookup.
            capital: Current account capital.
            lot_size: Contract lot size.

        Returns:
            (allowed, reason) tuple.
        """
        # Trading hours
        now = datetime.now(tz=IST).time()
        start = time.fromisoformat(self._sg.trading_start)
        end = time.fromisoformat(self._sg.trading_end)
        if now < start or now > end:
            return False, f"outside trading hours ({self._sg.trading_start}-{self._sg.trading_end})"

        # Daily loss limit
        if capital > 0 and self._daily_pnl / capital <= self._sg.daily_loss_limit_pct:
            return False, f"daily loss limit hit ({self._daily_pnl / capital:.2%})"

        # Compute prospective greeks from signal legs
        new_delta, new_vega, new_gamma = 0.0, 0.0, 0.0
        for leg in signal.legs:
            contract = chain.get_contract(leg.strike, OptionType(leg.option_type))
            if contract is None:
                return False, f"contract not found: {leg.strike} {leg.option_type}"
            multiplier = lot_size * leg.lots
            sign = 1.0 if leg.action.value == "BUY" else -1.0
            new_delta += sign * contract.delta * multiplier
            new_vega += sign * contract.vega * multiplier
            new_gamma += sign * contract.gamma * multiplier

        # Check vega exposure
        total_vega = abs(self._portfolio_vega + new_vega)
        if total_vega > self._vrp.max_vega_exposure:
            return False, f"vega limit exceeded ({total_vega:.0f} > {self._vrp.max_vega_exposure:.0f})"

        # Check gamma exposure
        total_gamma = abs(self._portfolio_gamma + new_gamma)
        if total_gamma > self._vrp.max_gamma_exposure:
            return False, f"gamma limit exceeded ({total_gamma:.0f} > {self._vrp.max_gamma_exposure:.0f})"

        # Check delta exposure
        total_delta = abs(self._portfolio_delta + new_delta)
        if total_delta > self._vrp.max_delta_abs:
            return False, f"delta limit exceeded ({total_delta:.2f} > {self._vrp.max_delta_abs:.2f})"

        return True, "ok"

    def update_greeks(
        self, chain: OptionChain, positions: list[dict], lot_size: int
    ) -> None:
        """Recalculate portfolio greeks from current positions.

        Args:
            chain: Current option chain.
            positions: List of position dicts with keys: strike, option_type, action, lots.
            lot_size: Contract lot size.
        """
        self._portfolio_delta = 0.0
        self._portfolio_vega = 0.0
        self._portfolio_gamma = 0.0

        for pos in positions:
            contract = chain.get_contract(pos["strike"], OptionType(pos["option_type"]))
            if contract is None:
                continue
            multiplier = lot_size * pos.get("lots", 1)
            sign = 1.0 if pos["action"] == "BUY" else -1.0
            self._portfolio_delta += sign * contract.delta * multiplier
            self._portfolio_vega += sign * contract.vega * multiplier
            self._portfolio_gamma += sign * contract.gamma * multiplier

    def on_trade_closed(self, pnl: float) -> None:
        self._daily_pnl += pnl

    def reset_daily(self) -> None:
        self._daily_pnl = 0.0
        self._portfolio_delta = 0.0
        self._portfolio_vega = 0.0
        self._portfolio_gamma = 0.0
