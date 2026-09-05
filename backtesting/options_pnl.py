"""Multi-leg options P&L calculator."""

from __future__ import annotations

from signals.base import OptionLeg, StrategyAction


def leg_pnl(
    action: StrategyAction,
    entry_price: float,
    exit_price: float,
    lots: int,
    lot_size: int,
) -> float:
    multiplier = lots * lot_size
    if action == StrategyAction.BUY:
        return (exit_price - entry_price) * multiplier
    return (entry_price - exit_price) * multiplier


def trade_pnl(
    legs: tuple[OptionLeg, ...],
    entry_prices: dict[str, float],
    exit_prices: dict[str, float],
    lot_size: int,
) -> float:
    total = 0.0
    for leg in legs:
        sym = leg.trading_symbol
        total += leg_pnl(
            leg.action,
            entry_prices[sym],
            exit_prices[sym],
            leg.lots,
            lot_size,
        )
    return total


def max_loss_straddle(
    ce_premium: float, pe_premium: float, lot_size: int
) -> float:
    """Reference max loss for a short straddle (theoretically unlimited).

    Returns total premium collected as a reference measure.
    """
    return (ce_premium + pe_premium) * lot_size
