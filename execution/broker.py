"""Broker wrapper for Groww order placement."""

from __future__ import annotations

import logging
import uuid

from growwapi import GrowwAPI

from config.settings import OrderType, Settings
from strategies.base import Signal

logger = logging.getLogger(__name__)


class Broker:
    """Wraps GrowwAPI for order placement with dry-run support."""

    def __init__(self, groww: GrowwAPI, settings: Settings) -> None:
        self._groww = groww
        self._settings = settings

    def place_order(
        self,
        signal: Signal,
        quantity: int,
        price: float | None = None,
    ) -> dict:
        """Place an order via Groww (or log mock in dry-run mode).

        Args:
            signal: Trade signal with direction and symbol.
            quantity: Number of shares.
            price: Reference price for limit orders. Ignored for market orders.

        Returns:
            Groww API response dict, or a mock dict if dry_run is enabled.
        """
        s = self._settings
        g = self._groww

        transaction_type = (
            g.TRANSACTION_TYPE_BUY
            if signal.direction == "BUY"
            else g.TRANSACTION_TYPE_SELL
        )

        # Determine order type and effective price
        order_type = g.ORDER_TYPE_MARKET
        limit_price = 0.0
        if s.trading.default_order_type == OrderType.LIMIT and price is not None:
            order_type = g.ORDER_TYPE_LIMIT
            buf = s.safeguards.slippage_buffer_pct
            if signal.direction == "BUY":
                limit_price = round(price * (1 + buf), 2)
            else:
                limit_price = round(price * (1 - buf), 2)

        # Use intraday product for scalping
        product = g.PRODUCT_INTRADAY

        order_ref = str(uuid.uuid4())

        params = dict(
            trading_symbol=signal.symbol,
            quantity=quantity,
            validity=g.VALIDITY_DAY,
            exchange=s.groww.exchange,
            segment=s.groww.segment,
            product=product,
            order_type=order_type,
            transaction_type=transaction_type,
            price=limit_price if order_type == g.ORDER_TYPE_LIMIT else 0,
            order_reference_id=order_ref,
        )

        if s.dry_run:
            logger.info("DRY RUN order: %s", params)
            return {"status": "SUCCESS", "dry_run": True, "params": params}

        return g.place_order(**params)

    def place_sl_order(self, signal: Signal, quantity: int, trigger_price: float) -> dict:
        """Place a stop-loss market order."""
        g = self._groww
        s = self._settings
        params = dict(
            trading_symbol=signal.symbol,
            quantity=quantity,
            validity=g.VALIDITY_DAY,
            exchange=s.groww.exchange,
            segment=s.groww.segment,
            product=g.PRODUCT_INTRADAY,
            order_type=g.ORDER_TYPE_STOP_LOSS_MARKET,
            transaction_type=g.TRANSACTION_TYPE_SELL,
            trigger_price=round(trigger_price, 2),
            price=0,
            order_reference_id=str(uuid.uuid4()),
        )
        if s.dry_run:
            logger.info("DRY RUN SL order: %s", params)
            return {"status": "SUCCESS", "dry_run": True, "params": params}
        return g.place_order(**params)

    def place_oco_order(
        self,
        symbol: str,
        quantity: int,
        tp_price: float,
        sl_trigger: float,
    ) -> dict:
        """Place an OCO smart order — target (LIMIT sell) + stop-loss (SL_M) as one unit.

        Args:
            symbol: Trading symbol.
            quantity: Position size to exit.
            tp_price: Take-profit price (used as both trigger and limit).
            sl_trigger: Stop-loss trigger price for SL_M leg.
        """
        g = self._groww
        s = self._settings
        ref = f"oco-{uuid.uuid4().hex[:12]}"
        params = dict(
            smart_order_type=g.SMART_ORDER_TYPE_OCO,
            reference_id=ref,
            segment=s.groww.segment,
            trading_symbol=symbol,
            quantity=quantity,
            net_position_quantity=quantity,
            transaction_type=g.TRANSACTION_TYPE_SELL,
            product_type=g.PRODUCT_INTRADAY,
            exchange=s.groww.exchange,
            duration=g.VALIDITY_DAY,
            target={
                "trigger_price": str(round(tp_price, 2)),
                "order_type": g.ORDER_TYPE_LIMIT,
                "price": str(round(tp_price, 2)),
            },
            stop_loss={
                "trigger_price": str(round(sl_trigger, 2)),
                "order_type": g.ORDER_TYPE_STOP_LOSS_MARKET,
                "price": None,
            },
        )
        if s.dry_run:
            logger.info("DRY RUN OCO order: %s", params)
            return {"status": "SUCCESS", "dry_run": True, "params": params}
        return g.create_smart_order(**params)
