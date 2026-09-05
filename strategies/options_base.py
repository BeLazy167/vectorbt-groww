"""Abstract base class for options strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from config.options_config import VRPConfig
from data.chain import OptionChain
from signals.base import OptionLeg, OptionsSignal
from signals.composite import VolSignalSnapshot


class OptionsStrategy(ABC):
    """Base class all options strategies must extend."""

    name: str

    @abstractmethod
    def generate_signal(
        self,
        chain: OptionChain,
        underlying_prices: pd.Series,
        vol_snapshot: VolSignalSnapshot,
        model_prediction: float | None = None,
    ) -> OptionsSignal | None:
        """Produce an entry signal from current market state."""

    @abstractmethod
    def construct_legs(
        self, chain: OptionChain, lot_size: int
    ) -> tuple[OptionLeg, ...]:
        """Build the option legs for this strategy."""

    @abstractmethod
    def exit_condition(
        self,
        chain: OptionChain,
        entry_signal: OptionsSignal,
        days_in_trade: int,
        current_pnl: float,
        vrp_config: VRPConfig,
    ) -> bool:
        """Check if position should be closed."""
