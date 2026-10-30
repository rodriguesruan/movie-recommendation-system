from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders import ensure_movie_lens_data, load_movie_lens_data
from src.validation.recommendation_service import MODEL_FEATURES, train_recommendation_model
from src.validation.temporal_strategy import (
    compute_global_mean_baseline,
    compute_regression_metrics,
    compute_user_item_bias_baseline,
    temporal_split_by_timestamp,
)


def evaluate_serving_model(data_dir: str | Path = 'data/raw/ml-1m') -> dict[str, dict[str, float]]:
    """Compare serving predictions with simple baselines on a strict time holdout."""
    resolved_data_dir = ensure_movie_lens_data(data_dir)
    ratings, _, _ = load_movie_lens_data(resolved_data_dir)
    train, validation = temporal_split_by_timestamp(ratings, validation_fraction=0.2)
    model = train_recommendation_model(train)

    user_stats = train.groupby('userId')['rating'].agg(['mean', 'count'])
    item_stats = train.groupby('movieId')['rating'].agg(['mean', 'count'])
    validation_features = validation[['userId', 'movieId']].copy()
    global_mean = float(train['rating'].mean())
    validation_features['user_mean_rating'] = (
        validation_features['userId'].map(user_stats['mean']).fillna(global_mean)
    )
    validation_features['user_rating_count'] = (
        validation_features['userId'].map(user_stats['count']).fillna(0)
    )
    validation_features['item_mean_rating'] = (
        validation_features['movieId'].map(item_stats['mean']).fillna(global_mean)
    )
    validation_features['item_rating_count'] = (
        validation_features['movieId'].map(item_stats['count']).fillna(0)
    )
    validation_features['user_item_interaction_count'] = 0

    predictions = model.predict(validation_features[MODEL_FEATURES])
    return {
        'global_mean': compute_regression_metrics(
            validation['rating'],
            compute_global_mean_baseline(train, validation),
        ),
        'user_item_bias': compute_regression_metrics(
            validation['rating'],
            compute_user_item_bias_baseline(train, validation),
        ),
        'random_forest_serving': compute_regression_metrics(
            validation['rating'],
            pd.Series(predictions, index=validation.index),
        ),
    }


def main() -> None:
    metrics = evaluate_serving_model('data/raw/ml-1m')
    print(f"{'Model':<24} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
    for name, result in metrics.items():
        print(
            f"{name:<24} {result['rmse']:>8.4f} "
            f"{result['mae']:>8.4f} {result['r2']:>8.4f}"
        )


if __name__ == '__main__':
    main()
