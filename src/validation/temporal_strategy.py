from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def temporal_split_by_timestamp(
    ratings_df: pd.DataFrame,
    validation_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings by time while preserving the temporal ordering of the data.

    The validation set is always built from the latest events, so no future
    information is allowed in the training set.
    """
    if 'timestamp' not in ratings_df.columns:
        raise KeyError("The ratings dataframe must contain a 'timestamp' column.")
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between 0 and 1.')

    sorted_df = ratings_df.sort_values('timestamp').reset_index(drop=True)
    split_index = max(1, min(len(sorted_df) - 1, int(len(sorted_df) * (1 - validation_fraction))))

    train_df = sorted_df.iloc[:split_index].copy()
    val_df = sorted_df.iloc[split_index:].copy()

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError('The temporal split produced an empty train or validation set.')

    return train_df, val_df


def leakage_audit(train_df: pd.DataFrame, val_df: pd.DataFrame) -> List[str]:
    """Return a list of leakage warnings for a time-based split.

    The audit only checks for obvious temporal contamination: validation rows must
    not include timestamps that are earlier than the most recent training event.
    """
    issues: List[str] = []

    if 'timestamp' not in train_df.columns or 'timestamp' not in val_df.columns:
        issues.append('Missing timestamp column in one of the splits.')
        return issues

    if val_df['timestamp'].min() < train_df['timestamp'].max():
        issues.append('Validation data overlaps with future information from the training period.')

    return issues


def compute_global_mean_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.Series:
    """Predict every validation rating with the training global mean."""
    if 'rating' not in train_df.columns:
        raise KeyError("The train dataframe must contain a 'rating' column.")

    global_mean = float(train_df['rating'].mean())
    return pd.Series(global_mean, index=val_df.index, dtype='float64')


def compute_user_item_bias_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.Series:
    """Compute a temporal-safe baseline using user and item biases.

    The biases are estimated from historical training data only and smoothed with
    a weak prior to reduce instability for cold-start users or movies.
    """
    if {'userId', 'movieId', 'rating'}.difference(train_df.columns):
        missing = {'userId', 'movieId', 'rating'} - set(train_df.columns)
        raise KeyError(f"The train dataframe is missing required columns: {sorted(missing)}")

    if {'userId', 'movieId'}.difference(val_df.columns):
        missing = {'userId', 'movieId'} - set(val_df.columns)
        raise KeyError(f"The validation dataframe is missing required columns: {sorted(missing)}")

    global_mean = float(train_df['rating'].mean())
    user_counts = train_df.groupby('userId').size()
    user_means = train_df.groupby('userId')['rating'].mean()
    item_counts = train_df.groupby('movieId').size()
    item_means = train_df.groupby('movieId')['rating'].mean()

    smoothing = 10.0
    user_bias = (
        (user_counts * user_means) + (smoothing * global_mean)
    ) / (user_counts + smoothing) - global_mean
    item_bias = (
        (item_counts * item_means) + (smoothing * global_mean)
    ) / (item_counts + smoothing) - global_mean

    predictions = (
        global_mean
        + val_df['userId'].map(user_bias).fillna(0.0)
        + val_df['movieId'].map(item_bias).fillna(0.0)
    )

    return predictions.astype(float)


def compute_regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute RMSE, MAE and R² for a rating prediction task."""
    y_true = pd.Series(y_true).astype(float).reset_index(drop=True)
    y_pred = pd.Series(y_pred).astype(float).reset_index(drop=True)

    residuals = y_true - y_pred
    mse = float(np.mean(residuals ** 2))
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if np.isclose(ss_tot, 0.0):
        r2 = 1.0 if np.isclose(ss_res, 0.0) else 0.0
    else:
        r2 = float(1.0 - (ss_res / ss_tot))

    return {'rmse': rmse, 'mae': mae, 'r2': r2}
