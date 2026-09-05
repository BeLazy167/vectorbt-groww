"""Prediction from trained VRP model."""

from __future__ import annotations

import pandas as pd

from config.options_config import ModelConfig
from models.features import _SNAPSHOT_FIELD_MAP
from models.pipeline import load_model
from signals.composite import VolSignalSnapshot


def extract_features_from_snapshot(
    snapshot: VolSignalSnapshot, config: ModelConfig
) -> pd.DataFrame:
    """Single-row DataFrame with feature columns from snapshot fields."""
    row = {}
    for feat in config.features:
        field_name = _SNAPSHOT_FIELD_MAP.get(feat, feat)
        row[feat] = getattr(snapshot, field_name)
    return pd.DataFrame([row], columns=config.features)


def predict_vrp(snapshot: VolSignalSnapshot, model_path: str, config: ModelConfig) -> float:
    """Load model and predict from a single snapshot."""
    result = load_model(model_path)
    features = extract_features_from_snapshot(snapshot, config)
    return float(result.model.predict(features)[0])
