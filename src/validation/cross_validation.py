from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd

from src.validation.model_baselines import TimestampAwareRandomForestRegressor
from src.validation.temporal_strategy import compute_regression_metrics


def make_temporal_folds(ratings_df: pd.DataFrame, n_splits: int = 3) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Create chronologically ordered train/validation folds without leakage."""
    if n_splits < 2:
        raise ValueError('n_splits must be at least 2.')
    if 'timestamp' not in ratings_df.columns:
        raise KeyError("The ratings dataframe must contain a 'timestamp' column.")

    sorted_df = ratings_df.sort_values('timestamp').reset_index(drop=True)
    if len(sorted_df) < n_splits + 1:
        raise ValueError('The dataset is too small for the requested number of temporal splits.')

    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    total_rows = len(sorted_df)
    for i in range(n_splits):
        start = i * total_rows // n_splits
        end = (i + 1) * total_rows // n_splits
        if start == end:
            continue
        train_end = start + max(1, (end - start) // 2)
        train_df = sorted_df.iloc[:train_end].copy()
        val_df = sorted_df.iloc[train_end:end].copy()
        if len(train_df) == 0 or len(val_df) == 0:
            continue
        folds.append((train_df, val_df))

    if not folds:
        raise ValueError('No valid temporal folds could be generated.')
    return folds


def optimize_random_forest(ratings_df: pd.DataFrame, n_trials: int = 3) -> dict[str, Any]:
    """Run a lightweight temporal hyperparameter search for the Random Forest baseline."""
    random.seed(42)
    np.random.seed(42)

    folds = make_temporal_folds(ratings_df, n_splits=3)
    candidate_params = [
        {'n_estimators': 50, 'max_depth': 5, 'min_samples_leaf': 1},
        {'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 1},
        {'n_estimators': 200, 'max_depth': 8, 'min_samples_leaf': 2},
    ]

    history: list[dict[str, Any]] = []
    best_rmse = np.inf
    best_params: dict[str, Any] | None = None

    for trial_index in range(min(n_trials, len(candidate_params))):
        params = candidate_params[trial_index]
        fold_scores: list[float] = []

        for train_df, val_df in folds:
            feature_cols = [col for col in train_df.columns if col not in {'rating', 'timestamp'}]
            model = TimestampAwareRandomForestRegressor(
                random_state=42,
                n_jobs=-1,
                **params,
            )
            model.fit(train_df[feature_cols], train_df['rating'])
            preds = model.predict(val_df[feature_cols])
            metrics = compute_regression_metrics(val_df['rating'], preds)
            fold_scores.append(metrics['rmse'])

        mean_rmse = float(np.mean(fold_scores))
        history.append({'trial': trial_index, 'params': params, 'cv_rmse': mean_rmse})
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = params

    if best_params is None:
        raise ValueError('No valid Random Forest configuration was evaluated.')

    return {
        'best_params': best_params,
        'cv_rmse': best_rmse,
        'history': history,
    }
