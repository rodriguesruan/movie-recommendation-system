from __future__ import annotations

import pandas as pd

from src.validation.temporal_strategy import temporal_split_by_timestamp


def build_user_movie_features(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Create rating-history features using events strictly before each row."""
    features = ratings_df.copy().sort_values('timestamp', kind='mergesort').reset_index(drop=True)

    user_groups = features.groupby('userId', sort=False)['rating']
    item_groups = features.groupby('movieId', sort=False)['rating']
    features['user_rating_count'] = user_groups.cumcount()
    features['item_rating_count'] = item_groups.cumcount()

    user_prior_sum = user_groups.cumsum() - features['rating']
    item_prior_sum = item_groups.cumsum() - features['rating']
    features['user_mean_rating'] = (
        user_prior_sum.div(features['user_rating_count'].replace(0, pd.NA)).fillna(3.0)
    )
    features['item_mean_rating'] = (
        item_prior_sum.div(features['item_rating_count'].replace(0, pd.NA)).fillna(3.0)
    )
    features['user_item_interaction_count'] = (
        features.groupby(['userId', 'movieId'], sort=False).cumcount()
    )

    return features


def make_temporal_split_features(
    ratings_df: pd.DataFrame,
    validation_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build historical features before splitting on time, then split chronologically."""
    prepared = build_user_movie_features(ratings_df)
    train_df, val_df = temporal_split_by_timestamp(prepared, validation_fraction=validation_fraction)

    train_df = train_df.copy()
    val_df = val_df.copy()

    user_mean_global = train_df['user_mean_rating'].mean()
    item_mean_global = train_df['item_mean_rating'].mean()
    train_df['user_mean_rating'] = train_df['user_mean_rating'].fillna(user_mean_global)
    train_df['item_mean_rating'] = train_df['item_mean_rating'].fillna(item_mean_global)
    val_df['user_mean_rating'] = val_df['user_mean_rating'].fillna(user_mean_global)
    val_df['item_mean_rating'] = val_df['item_mean_rating'].fillna(item_mean_global)

    return train_df, val_df
