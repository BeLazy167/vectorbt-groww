"""Pre-order safeguard checks."""

from __future__ import annotations

import logging
from datetime import datetime, time

from zoneinfo import ZoneInfo

from config.settings import Settings
from strategies.base import Signal

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")


class SafeguardChecker:
    """Validates trading conditions before order placement."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._daily_pnl: float = 0.0
        self._open_positions: int = 0
        self._candles_since_loss: int = 0
        self._last_loss: bool = False

    def check(self, signal: Signal, capital: float) -> tuple[bool, str]:
        """Run all safeguard checks in priority order.

        Args:
            signal: Incoming trade signal.
            capital: Current account capital for ratio calculations.

        Returns:
            (allowed, reason) -- True/"ok" if all checks pass, else
            False with rejection reason.
        """
        sg = self._settings.safeguards

        # 1. Trading hours
        now = datetime.now(tz=IST).time()
        start = time.fromisoformat(sg.trading_start)
        end = time.fromisoformat(sg.trading_end)
        if now < start or now > end:
            return False, f"outside trading hours ({sg.trading_start}-{sg.trading_end})"

        # 2. Daily loss limit
        if capital > 0 and self._daily_pnl / capital <= sg.daily_loss_limit_pct:
            return False, f"daily loss limit hit ({self._daily_pnl / capital:.2%})"

        # 3. Max open positions
        if self._open_positions >= sg.max_open_positions:
            return False, f"max open positions reached ({sg.max_open_positions})"

        # 4. Cooldown after loss
        if self._last_loss and self._candles_since_loss < sg.cooldown_candles:
            remaining = sg.cooldown_candles - self._candles_since_loss
            return False, f"cooldown active ({remaining} candles remaining)"

        return True, "ok"

    def on_candle_close(self) -> None:
        """Increment cooldown counter if in cooldown."""
        if self._last_loss:
            self._candles_since_loss += 1

    def on_trade_closed(self, pnl: float) -> None:
        """Update daily PnL and loss tracking."""
        self._daily_pnl += pnl
        self._last_loss = pnl < 0
        self._candles_since_loss = 0

    def on_position_opened(self) -> None:
        self._open_positions += 1

    def on_position_closed(self) -> None:
        self._open_positions = max(0, self._open_positions - 1)

    def reset_daily(self) -> None:
        """Reset all daily state (call at start of trading day)."""
        self._daily_pnl = 0.0
        self._candles_since_loss = 0
        self._last_loss = False
