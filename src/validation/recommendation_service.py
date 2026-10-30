from __future__ import annotations

from typing import Any

import pandas as pd

from src.validation.feature_engineering import build_user_movie_features
from src.validation.final_recommendations import build_recommendation_candidates
from src.validation.model_baselines import TimestampAwareRandomForestRegressor


MODEL_FEATURES = [
    'userId',
    'movieId',
    'user_mean_rating',
    'user_rating_count',
    'item_mean_rating',
    'item_rating_count',
    'user_item_interaction_count',
]
DEFAULT_RATING = 3.0
MAX_TRAINING_ROWS = 30_000


def train_recommendation_model(
    ratings_df: pd.DataFrame,
    max_training_rows: int = MAX_TRAINING_ROWS,
) -> TimestampAwareRandomForestRegressor:
    """Train a serving model on historical features without target leakage."""
    if ratings_df.empty:
        raise ValueError('Ratings data cannot be empty.')
    if max_training_rows < 1:
        raise ValueError('max_training_rows must be at least 1.')

    training_ratings = ratings_df
    if len(training_ratings) > max_training_rows:
        training_ratings = training_ratings.sample(
            n=max_training_rows,
            random_state=42,
        ).sort_values('timestamp')
    training = build_user_movie_features(training_ratings)

    model = TimestampAwareRandomForestRegressor(
        n_estimators=40,
        max_depth=14,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=1,
    )
    model.fit(training[MODEL_FEATURES], training['rating'])
    return model


def generate_recommendations_for_user(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    user_id: int,
    top_n: int = 10,
    model: Any | None = None,
) -> list[dict]:
    """Return unseen movies ranked by the trained model with historical context."""
    if top_n < 1:
        raise ValueError('top_n must be at least 1.')
    if user_id not in set(ratings_df['userId'].unique()):
        return []

    candidates = build_recommendation_candidates(
        ratings_df,
        movies_df,
        user_id=user_id,
        top_n=len(movies_df),
    )
    if candidates.empty:
        return []

    if model is None:
        model = train_recommendation_model(ratings_df)

    user_ratings = ratings_df.loc[ratings_df['userId'] == user_id, 'rating']
    user_mean = float(user_ratings.mean()) if not user_ratings.empty else DEFAULT_RATING
    item_stats = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count'])

    candidates = candidates.copy()
    candidates['userId'] = user_id
    candidates['user_mean_rating'] = user_mean
    candidates['user_rating_count'] = len(user_ratings)
    candidates['item_mean_rating'] = candidates['movieId'].map(item_stats['mean']).fillna(DEFAULT_RATING)
    candidates['item_rating_count'] = candidates['movieId'].map(item_stats['count']).fillna(0).astype(int)
    candidates['user_item_interaction_count'] = 0

    feature_columns = list(getattr(model, 'feature_names_in_', MODEL_FEATURES))
    predictions = model.predict(candidates.reindex(columns=feature_columns))
    candidates['prediction'] = pd.Series(predictions, index=candidates.index).clip(0.5, 5.0)
    candidates = candidates.sort_values(
        ['prediction', 'movieId'], ascending=[False, True]
    ).head(top_n)

    results = []
    for row in candidates.itertuples(index=False):
        item_mean = float(row.item_mean_rating)
        item_count = int(row.item_rating_count)
        item_context = (
            f"This movie's historical average rating is {item_mean:.2f} from {item_count} ratings."
            if item_count
            else 'This movie has no rating history in the dataset yet.'
        )
        explanation = (
            f"Your {len(user_ratings)} previous ratings average {user_mean:.2f}. "
            f"{item_context} The model predicts {float(row.prediction):.2f} out of 5."
        )
        results.append(
            {
                'movieId': int(row.movieId),
                'title': str(row.title),
                'genres': str(row.genres),
                'prediction': float(row.prediction),
                'score': float(row.prediction),
                'explanation': explanation,
            }
        )

    return results
