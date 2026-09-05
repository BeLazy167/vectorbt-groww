"""Multi-leg options order placement via Groww."""

from __future__ import annotations

import logging
import uuid

from growwapi import GrowwAPI

from config.settings import OrderType, Settings
from signals.base import OptionsSignal, StrategyAction

logger = logging.getLogger(__name__)


class OptionsBroker:
    """Places individual legs of a multi-leg options order."""

    def __init__(self, groww: GrowwAPI, settings: Settings) -> None:
        self._groww = groww
        self._settings = settings

    def place_legs(
        self, signal: OptionsSignal, lot_size: int
    ) -> list[dict]:
        """Place each leg of the options signal as a separate order.

        Args:
            signal: Options signal with legs to execute.
            lot_size: Contract lot size for the underlying.

        Returns:
            List of order response dicts (one per leg). Each dict includes
            a "leg" key with the trading_symbol for identification.
        """
        results = []
        for leg in signal.legs:
            try:
                result = self._place_single_leg(leg, lot_size)
                result["leg"] = leg.trading_symbol
                results.append(result)
            except Exception:
                logger.exception("Failed to place leg %s", leg.trading_symbol)
                results.append({"status": "FAILED", "leg": leg.trading_symbol})
                if any(r.get("status") == "SUCCESS" for r in results[:-1]):
                    logger.error(
                        "PARTIAL FILL: %d/%d legs placed for %s. Manual intervention required.",
                        sum(1 for r in results if r.get("status") == "SUCCESS"),
                        len(signal.legs),
                        signal.underlying,
                    )
                break
        return results

    def _place_single_leg(self, leg, lot_size: int) -> dict:
        s = self._settings
        g = self._groww

        transaction_type = (
            g.TRANSACTION_TYPE_BUY
            if leg.action == StrategyAction.BUY
            else g.TRANSACTION_TYPE_SELL
        )

        quantity = leg.lots * lot_size
        order_ref = str(uuid.uuid4())

        params = dict(
            trading_symbol=leg.trading_symbol,
            quantity=quantity,
            validity=g.VALIDITY_DAY,
            exchange=s.groww.exchange,
            segment="FNO",
            product=g.PRODUCT_INTRADAY,
            order_type=g.ORDER_TYPE_MARKET,
            transaction_type=transaction_type,
            price=0,
            order_reference_id=order_ref,
        )

        if s.dry_run:
            logger.info("DRY RUN F&O order: %s", params)
            return {"status": "SUCCESS", "dry_run": True, "params": params}

        return g.place_order(**params)
