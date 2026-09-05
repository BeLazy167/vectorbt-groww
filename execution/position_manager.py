"""Position tracking and unrealized PnL computation."""

from __future__ import annotations

import logging

from growwapi import GrowwAPI

from config.settings import Settings

logger = logging.getLogger(__name__)


class PositionManager:
    """Tracks open positions and computes unrealized PnL via Groww API."""

    def __init__(self, groww: GrowwAPI, settings: Settings) -> None:
        self._groww = groww
        self._settings = settings
        self._positions: dict[str, dict] = {}

    def sync_positions(self) -> None:
        """Fetch positions from Groww and update local cache.

        Calls groww.get_positions and parses the response payload
        into an internal dict keyed by trading_symbol.
        """
        resp = self._groww.get_positions(segment=self._settings.groww.segment)
        if resp.get("status") != "SUCCESS":
            logger.warning("Failed to sync positions: %s", resp)
            return

        positions = resp.get("payload", {}).get("positions", [])
        self._positions = {p["trading_symbol"]: p for p in positions}
        logger.debug("Synced %d positions", len(self._positions))

    def get_position(self, symbol: str) -> dict | None:
        return self._positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def get_unrealized_pnl(self, symbol: str, current_price: float) -> float:
        """Compute unrealized PnL for a single position.

        Args:
            symbol: Trading symbol.
            current_price: Latest market price.

        Returns:
            Unrealized PnL value. 0.0 if no position found.
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return 0.0
        qty = pos.get("quantity", 0)
        net_price = pos.get("net_price", 0.0)
        return (current_price - net_price) * qty

    def get_total_unrealized_pnl(self, current_prices: dict[str, float]) -> float:
        """Sum unrealized PnL across all cached positions.

        Args:
            current_prices: Map of symbol to current market price.

        Returns:
            Total unrealized PnL.
        """
        total = 0.0
        for symbol, pos in self._positions.items():
            price = current_prices.get(symbol)
            if price is not None:
                qty = pos.get("quantity", 0)
                net_price = pos.get("net_price", 0.0)
                total += (price - net_price) * qty
        return total
