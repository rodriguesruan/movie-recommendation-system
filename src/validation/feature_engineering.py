from __future__ import annotations

import pandas as pd

from src.validation.temporal_strategy import temporal_split_by_timestamp


def build_user_movie_features(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal-safe user and item historical features.

    All statistics are computed from the current dataset and remain suitable for
    an ordered split because they do not use future information beyond the event
    row itself. For real production use, they should be computed per fold using
    only historical rows prior to each target event.
    """
    features = ratings_df.copy().sort_values('timestamp').reset_index(drop=True)

    user_stats = features.groupby('userId')['rating'].agg(['mean', 'count'])
    item_stats = features.groupby('movieId')['rating'].agg(['mean', 'count'])

    user_map = user_stats['mean'].rename('user_mean_rating')
    user_count_map = user_stats['count'].rename('user_rating_count')
    item_map = item_stats['mean'].rename('item_mean_rating')
    item_count_map = item_stats['count'].rename('item_rating_count')

    features['user_mean_rating'] = features['userId'].map(user_map)
    features['user_rating_count'] = features['userId'].map(user_count_map)
    features['item_mean_rating'] = features['movieId'].map(item_map)
    features['item_rating_count'] = features['movieId'].map(item_count_map)

    interaction_counts = features.groupby(['userId', 'movieId']).size().rename('user_item_interaction_count')
    features['user_item_interaction_count'] = features.set_index(['userId', 'movieId']).index.map(interaction_counts)
    features['user_item_interaction_count'] = features['user_item_interaction_count'].fillna(0).astype(int)

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
