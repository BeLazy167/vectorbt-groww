"""Model training, walk-forward backtest, and persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelResult:
    model: Any
    scaler: StandardScaler
    feature_names: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    predictions: pd.Series | None = None


def train_model(
    features: pd.DataFrame, target: pd.Series, model_type: str = "regression"
) -> ModelResult:
    """Train a gradient boosting model with time-series CV.

    Args:
        features: Feature matrix.
        target: Target series aligned with features.
        model_type: "regression" or "classification".

    Returns:
        ModelResult with fitted model, scaler, and metrics.
    """
    scaler = StandardScaler()
    if model_type == "classification":
        estimator = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    else:
        estimator = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)

    pipe = Pipeline([("scaler", scaler), ("model", estimator)])
    tscv = TimeSeriesSplit(n_splits=5)

    scoring = "r2" if model_type == "regression" else "accuracy"
    cv_scores = cross_val_score(pipe, features, target, cv=tscv, scoring=scoring)

    pipe.fit(features, target)
    preds = pd.Series(pipe.predict(features), index=features.index)

    if model_type == "regression":
        metrics = {
            "r2": r2_score(target, preds),
            "mse": mean_squared_error(target, preds),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }
    else:
        metrics = {
            "accuracy": accuracy_score(target, preds),
            "f1": f1_score(target, preds),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }

    return ModelResult(
        model=pipe,
        scaler=pipe.named_steps["scaler"],
        feature_names=list(features.columns),
        metrics=metrics,
        predictions=preds,
    )


def walk_forward_backtest(
    features: pd.DataFrame, target: pd.Series, window: int = 60
) -> pd.DataFrame:
    """Expanding-window walk-forward backtest.

    Args:
        features: Feature matrix with datetime index.
        target: Target series aligned with features.
        window: Minimum training window size.

    Returns:
        DataFrame with columns [actual, predicted, date].
    """
    results = []
    for i in range(window, len(features)):
        train_X = features.iloc[:i]
        train_y = target.iloc[:i]
        test_X = features.iloc[i : i + 1]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)),
        ])
        pipe.fit(train_X, train_y)
        pred = pipe.predict(test_X)[0]

        results.append({
            "date": features.index[i],
            "actual": target.iloc[i],
            "predicted": pred,
        })

    return pd.DataFrame(results)


def save_model(result: ModelResult, path: str) -> None:
    joblib.dump(result, path)


def load_model(path: str) -> ModelResult:
    return joblib.load(path)
