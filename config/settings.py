"""Central configuration for the trading pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class Segment(Enum):
    EQUITY = "EQUITY"
    FNO = "FNO"


@dataclass
class GrowwConfig:
    """Auth: api_key + secret (checksum) or api_key + totp."""
    api_key: str = field(default_factory=lambda: os.environ.get("GROWW_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.environ.get("GROWW_API_SECRET", ""))
    totp_secret: str = field(default_factory=lambda: os.environ.get("GROWW_TOTP_SECRET", ""))
    exchange: str = "NSE"
    segment: str = "CASH"

    _cached_token: str = ""

    def get_token(self) -> str:
        """Get access token via secret or TOTP fallback. Cached per session."""
        if self._cached_token:
            return self._cached_token
        from growwapi import GrowwAPI

        if self.api_secret:
            self._cached_token = GrowwAPI.get_access_token(
                api_key=self.api_key, secret=self.api_secret
            )
        else:
            import pyotp
            totp = pyotp.TOTP(self.totp_secret).now()
            self._cached_token = GrowwAPI.get_access_token(
                api_key=self.api_key, totp=totp
            )
        return self._cached_token


@dataclass
class SafeguardConfig:
    max_position_pct: float = 0.20        # 20% of capital per position (1L on 5L)
    max_open_positions: int = 5            # 5 concurrent positions
    daily_loss_limit_pct: float = -0.02   # -2% daily loss limit
    cooldown_candles: int = 5             # candles to wait after loss
    trading_start: str = "09:20"          # IST
    trading_end: str = "15:10"            # IST
    slippage_buffer_pct: float = 0.0005   # 0.05% for limit orders


@dataclass
class GrowwDeliveryFees:
    """Groww equity delivery charges (NSE). See https://groww.in/pricing."""
    brokerage_flat: float = 20.0          # ₹20 per order (or 0.1%, whichever lower)
    brokerage_pct: float = 0.001          # 0.1%
    stt_pct: float = 0.001               # 0.1% both sides
    stamp_duty_buy_pct: float = 0.00015   # 0.015% buy only
    exchange_txn_pct: float = 0.0000297   # NSE
    sebi_pct: float = 0.000001            # 0.0001%
    ipft_pct: float = 0.000001            # 0.0001%
    dp_flat: float = 20.0                 # ₹20 per sell
    gst_pct: float = 0.18                 # 18% on applicable

    def effective_fees(self, trade_value: float = 100_000) -> tuple[float, float]:
        """Return (pct_per_side, fixed_per_side) for vectorbt.

        Averages asymmetric buy/sell components into symmetric values
        since vbt applies fees equally on both sides.
        """
        # Percentage components (applied both sides unless noted)
        pct_both = self.stt_pct + self.exchange_txn_pct + self.sebi_pct + self.ipft_pct
        gst_on_pct = (self.exchange_txn_pct + self.sebi_pct + self.ipft_pct) * self.gst_pct

        buy_pct = pct_both + self.stamp_duty_buy_pct + gst_on_pct
        sell_pct = pct_both + gst_on_pct
        avg_pct = (buy_pct + sell_pct) / 2

        # Flat components
        brokerage = min(self.brokerage_flat, self.brokerage_pct * trade_value)
        gst_on_brokerage = brokerage * self.gst_pct
        buy_flat = brokerage + gst_on_brokerage
        sell_flat = brokerage + gst_on_brokerage + self.dp_flat + self.dp_flat * self.gst_pct
        avg_flat = (buy_flat + sell_flat) / 2

        return avg_pct, avg_flat


@dataclass
class TradingConfig:
    capital: float = 500_000.0
    default_order_type: OrderType = OrderType.LIMIT
    fees: GrowwDeliveryFees = field(default_factory=GrowwDeliveryFees)
    default_timeframes: list[str] = field(default_factory=lambda: ["1m", "5m", "15m"])
    candle_buffer_size: int = 500         # rolling buffer per TF


@dataclass
class Settings:
    groww: GrowwConfig = field(default_factory=GrowwConfig)
    safeguards: SafeguardConfig = field(default_factory=SafeguardConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    dry_run: bool = True                  # paper mode by default
    log_level: str = "INFO"

    # Options trading configs (lazy import to avoid circular deps)
    @property
    def options(self):
        from config.options_config import OptionsConfig
        if not hasattr(self, "_options"):
            self._options = OptionsConfig()
        return self._options

    @property
    def vrp(self):
        from config.options_config import VRPConfig
        if not hasattr(self, "_vrp"):
            self._vrp = VRPConfig()
        return self._vrp

    @property
    def model(self):
        from config.options_config import ModelConfig
        if not hasattr(self, "_model"):
            self._model = ModelConfig()
        return self._model
