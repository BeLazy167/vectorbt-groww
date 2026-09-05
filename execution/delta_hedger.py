"""Periodic delta neutralization for options positions."""

from __future__ import annotations

import logging
import uuid

from growwapi import GrowwAPI

from config.settings import Settings
from config.options_config import VRPConfig
from data.chain import OptionChain, OptionContract
from signals.base import OptionType

logger = logging.getLogger(__name__)


class DeltaHedger:
    """Monitors portfolio delta and places hedges when threshold is breached."""

    def __init__(
        self, groww: GrowwAPI, settings: Settings, vrp_config: VRPConfig
    ) -> None:
        self._groww = groww
        self._settings = settings
        self._vrp = vrp_config

    def compute_portfolio_delta(
        self,
        chain: OptionChain,
        positions: list[dict],
        lot_size: int,
    ) -> float:
        """Sum net delta across all positions.

        Args:
            chain: Current option chain for greeks.
            positions: List of dicts with strike, option_type, action, lots.
            lot_size: Contract lot size.
        """
        net_delta = 0.0
        for pos in positions:
            contract = chain.get_contract(pos["strike"], OptionType(pos["option_type"]))
            if contract is None:
                continue
            multiplier = lot_size * pos.get("lots", 1)
            sign = 1.0 if pos["action"] == "BUY" else -1.0
            net_delta += sign * contract.delta * multiplier
        return net_delta

    def check_and_hedge(
        self,
        chain: OptionChain,
        positions: list[dict],
        lot_size: int,
    ) -> dict | None:
        """Check if delta hedge is needed and place hedge order.

        Args:
            chain: Current option chain.
            positions: Current option positions.
            lot_size: Contract lot size.

        Returns:
            Hedge order result dict, or None if no hedge needed.
        """
        net_delta = self.compute_portfolio_delta(chain, positions, lot_size)

        if abs(net_delta) <= self._vrp.delta_hedge_threshold:
            return None

        # Hedge via underlying futures
        # Negative delta -> buy futures, positive delta -> sell futures
        raw_lots = abs(net_delta) / lot_size
        rounded_lots = max(1, round(raw_lots))
        hedge_quantity = rounded_lots * lot_size
        if hedge_quantity == 0:
            return None

        transaction_type = (
            self._groww.TRANSACTION_TYPE_BUY
            if net_delta < 0
            else self._groww.TRANSACTION_TYPE_SELL
        )

        order_ref = str(uuid.uuid4())
        params = dict(
            trading_symbol=chain.underlying,
            quantity=hedge_quantity,
            validity=self._groww.VALIDITY_DAY,
            exchange=self._settings.groww.exchange,
            segment="FNO",
            product=self._groww.PRODUCT_INTRADAY,
            order_type=self._groww.ORDER_TYPE_MARKET,
            transaction_type=transaction_type,
            price=0,
            order_reference_id=order_ref,
        )

        if self._settings.dry_run:
            logger.info(
                "DRY RUN delta hedge: net_delta=%.2f, %s", net_delta, params
            )
            return {"status": "SUCCESS", "dry_run": True, "params": params}

        logger.info("Delta hedge: net_delta=%.2f, placing %s", net_delta, params)
        return self._groww.place_order(**params)
