"""Feature matrix construction from vol signal snapshots."""

from __future__ import annotations

import pandas as pd

from config.options_config import ModelConfig
from signals.composite import VolSignalSnapshot

_SNAPSHOT_FIELD_MAP = {
    "ivrv_ratio": "ivrv_ratio",
    "iv_percentile": "iv_percentile_rank",
    "rv_30d": "rv_30d",
    "flat_forward_ratio": "flat_forward_ratio",
    "days_to_expiry": "days_to_expiry",
    "atm_iv": "atm_iv",
}


def build_feature_matrix(
    snapshots: list[VolSignalSnapshot], config: ModelConfig
) -> pd.DataFrame:
    """Extract feature columns from snapshots into DataFrame."""
    rows = []
    for snap in snapshots:
        row = {}
        for feat in config.features:
            field_name = _SNAPSHOT_FIELD_MAP.get(feat, feat)
            row[feat] = getattr(snap, field_name)
        rows.append(row)
    return pd.DataFrame(rows, columns=config.features)


def build_target(straddle_pnls: pd.Series, model_type: str = "regression") -> pd.Series:
    """Build target series. Regression: raw PnL. Classification: sign."""
    if model_type == "classification":
        return (straddle_pnls > 0).astype(int)
    return straddle_pnls
