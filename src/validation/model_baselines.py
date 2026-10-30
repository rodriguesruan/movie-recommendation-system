from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

from src.validation.temporal_strategy import compute_regression_metrics


class TimestampAwareRandomForestRegressor(RandomForestRegressor):
    """Random Forest regressor that ignores non-feature temporal metadata."""

    def _prepare_features(self, X: Any) -> Any:
        if isinstance(X, pd.DataFrame):
            return X.drop(columns=[col for col in ['timestamp'] if col in X.columns], errors='ignore')
        return X

    def fit(self, X: Any, y: Any, *args, **kwargs):
        X_clean = self._prepare_features(X)
        return super().fit(X_clean, y, *args, **kwargs)

    def predict(self, X: Any, *args, **kwargs):
        X_clean = self._prepare_features(X)
        return super().predict(X_clean, *args, **kwargs)


def build_random_forest_model(train_df: pd.DataFrame, target: str = 'rating') -> TimestampAwareRandomForestRegressor:
    """Train a Random Forest regressor over the temporal-safe feature set."""
    if target not in train_df.columns:
        raise KeyError(f"Target column '{target}' not found in the training data.")

    feature_columns = [
        col for col in train_df.columns if col not in {target, 'timestamp'}
    ]

    model = TimestampAwareRandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_df[feature_columns], train_df[target])
    return model


def evaluate_model_on_split(y_true: pd.Series, y_pred: Any) -> dict[str, float]:
    """Compute standard regression metrics for a model prediction vector."""
    return compute_regression_metrics(y_true, pd.Series(y_pred))
