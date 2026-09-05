"""Options trading configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OptionsConfig:
    """General options trading parameters."""

    risk_free_rate: float = 0.065  # India 10Y
    iv_lookback_days: int = 252
    rv_window: int = 30
    min_option_volume: int = 1000
    min_oi: int = 5000
    days_before_expiry_entry: int = 7
    days_before_expiry_exit: int = 1


@dataclass
class VRPConfig:
    """Variance Risk Premium strategy thresholds."""

    ivrv_entry_threshold: float = 1.15
    iv_percentile_entry_min: float = 30.0
    iv_percentile_entry_max: float = 85.0
    max_vega_exposure: float = 50_000.0
    max_delta_abs: float = 0.3
    butterfly_wing_width: int = 2
    max_gamma_exposure: float = 10_000.0
    delta_hedge_threshold: float = 0.5
    profit_target_pct: float = 0.50  # exit at 50% of max premium
    stop_loss_pct: float = 2.0  # exit at 2x premium loss


@dataclass
class ModelConfig:
    """sklearn model configuration."""

    features: list[str] = field(
        default_factory=lambda: [
            "ivrv_ratio",
            "iv_percentile",
            "rv_30d",
            "flat_forward_ratio",
            "days_to_expiry",
        ]
    )
    target: str = "forward_straddle_pnl"
    test_size: float = 0.2
    walk_forward_window: int = 60
    model_type: str = "regression"  # "regression" or "classification"
