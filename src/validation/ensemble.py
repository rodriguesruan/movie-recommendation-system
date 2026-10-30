from __future__ import annotations

import numpy as np
import pandas as pd


def build_ensemble_predictions(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Construct a simple ensemble of base predictors from historical rating signals."""
    df = ratings_df.copy()
    global_mean = float(df['rating'].mean()) if 'rating' in df.columns else 0.0

    df['global_mean'] = global_mean
    df['user_bias'] = df.get('user_mean_rating', global_mean).fillna(global_mean)
    df['item_bias'] = df.get('item_mean_rating', global_mean).fillna(global_mean)

    if 'user_mean_rating' in df.columns and 'item_mean_rating' in df.columns:
        df['ensemble'] = 0.5 * df['user_mean_rating'] + 0.5 * df['item_mean_rating']
    else:
        df['ensemble'] = df['global_mean']

    return df


def train_blending_model(df: pd.DataFrame) -> dict[str, Any]:
    """Fit a tiny ridge-like blending model with a linear combination of base predictions."""
    import numpy as np

    feature_cols = [c for c in ['global_mean', 'user_bias', 'item_bias'] if c in df.columns]
    if not feature_cols:
        raise ValueError('At least one base prediction column is required for blending.')

    X = df[feature_cols].to_numpy(dtype=float)
    y = df['rating'].to_numpy(dtype=float)

    if X.shape[0] == 0:
        raise ValueError('The blending dataset is empty.')

    weights = np.linalg.lstsq(X, y, rcond=None)[0]
    intercept = float(np.mean(y - X @ weights))

    return {'weights': weights.tolist(), 'intercept': intercept}
