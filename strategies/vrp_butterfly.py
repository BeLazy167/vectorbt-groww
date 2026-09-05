"""VRP iron butterfly strategy."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from config.options_config import OptionsConfig, VRPConfig
from data.chain import OptionChain
from signals.base import OptionLeg, OptionType, OptionsSignal, StrategyAction
from signals.composite import VolSignalSnapshot, is_vrp_entry
from strategies.options_base import OptionsStrategy
from strategies.registry import register_options


def _NullContract() -> SimpleNamespace:
    return SimpleNamespace(ltp=0.0)


def _infer_strike_gap(chain: OptionChain) -> float:
    """Infer minimum strike gap from chain contracts."""
    strikes = sorted({c.strike for c in chain.contracts})
    if len(strikes) < 2:
        return 50.0
    gaps = [strikes[i + 1] - strikes[i] for i in range(len(strikes) - 1)]
    return min(gaps)


@register_options
class VRPButterflyStrategy(OptionsStrategy):
    """Iron butterfly exploiting variance risk premium with defined risk."""

    name = "vrp_butterfly"

    def __init__(
        self, options_config: OptionsConfig, vrp_config: VRPConfig
    ) -> None:
        self.options_config = options_config
        self.vrp_config = vrp_config

    def generate_signal(
        self,
        chain: OptionChain,
        underlying_prices: pd.Series,
        vol_snapshot: VolSignalSnapshot,
        model_prediction: float | None = None,
    ) -> OptionsSignal | None:
        if not is_vrp_entry(vol_snapshot, self.vrp_config):
            return None
        if model_prediction is not None and model_prediction <= 0:
            return None

        lot_size = 1
        legs = self.construct_legs(chain, lot_size)
        strength = min(vol_snapshot.ivrv_ratio / self.vrp_config.ivrv_entry_threshold, 1.0)

        return OptionsSignal(
            underlying=chain.underlying,
            expiry=chain.expiry,
            legs=legs,
            strategy_name=self.name,
            signal_strength=strength,
            entry_iv=vol_snapshot.atm_iv,
            entry_ivrv=vol_snapshot.ivrv_ratio,
            model_score=model_prediction,
        )

    def construct_legs(
        self, chain: OptionChain, lot_size: int
    ) -> tuple[OptionLeg, ...]:
        atm = chain.atm_strike()
        strike_gap = _infer_strike_gap(chain)
        wing_offset = self.vrp_config.butterfly_wing_width * strike_gap

        lower_strike = atm - wing_offset
        upper_strike = atm + wing_offset

        def _symbol(strike: float, ot: OptionType) -> str:
            contract = chain.get_contract(strike, ot)
            if contract:
                return contract.trading_symbol
            return f"{chain.underlying}-{chain.expiry}-{strike}-{ot.value}"

        return (
            OptionLeg(
                trading_symbol=_symbol(lower_strike, OptionType.PE),
                strike=lower_strike,
                option_type=OptionType.PE,
                action=StrategyAction.BUY,
                lots=lot_size,
            ),
            OptionLeg(
                trading_symbol=_symbol(atm, OptionType.PE),
                strike=atm,
                option_type=OptionType.PE,
                action=StrategyAction.SELL,
                lots=lot_size,
            ),
            OptionLeg(
                trading_symbol=_symbol(atm, OptionType.CE),
                strike=atm,
                option_type=OptionType.CE,
                action=StrategyAction.SELL,
                lots=lot_size,
            ),
            OptionLeg(
                trading_symbol=_symbol(upper_strike, OptionType.CE),
                strike=upper_strike,
                option_type=OptionType.CE,
                action=StrategyAction.BUY,
                lots=lot_size,
            ),
        )

    def exit_condition(
        self,
        chain: OptionChain,
        entry_signal: OptionsSignal,
        days_in_trade: int,
        current_pnl: float,
        vrp_config: VRPConfig,
    ) -> bool:
        expiry_date = pd.Timestamp(entry_signal.expiry)
        days_to_expiry = (expiry_date - pd.Timestamp(datetime.now().date())).days
        if days_to_expiry <= self.options_config.days_before_expiry_exit:
            return True

        entry_premium = sum(
            (chain.get_contract(leg.strike, leg.option_type) or _NullContract()).ltp
            for leg in entry_signal.legs
            if leg.action == StrategyAction.SELL
        )
        if entry_premium > 0:
            if current_pnl > vrp_config.profit_target_pct * entry_premium:
                return True
            if current_pnl < -vrp_config.stop_loss_pct * entry_premium:
                return True

        return False
