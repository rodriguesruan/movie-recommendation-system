import pandas as pd

from src.validation.feature_engineering import (
    build_user_movie_features,
    make_temporal_split_features,
)


def test_user_movie_feature_builder_creates_approved_columns():
    ratings = pd.DataFrame(
        {
            'userId': [1, 1, 2, 2],
            'movieId': [10, 11, 10, 12],
            'rating': [4.0, 5.0, 3.0, 2.0],
            'timestamp': [100, 200, 150, 250],
        }
    )

    features = build_user_movie_features(ratings)

    required = {
        'userId', 'movieId', 'rating', 'timestamp',
        'user_mean_rating', 'user_rating_count',
        'item_mean_rating', 'item_rating_count',
        'user_item_interaction_count'
    }
    assert required.issubset(features.columns)
    assert features['user_mean_rating'].notna().all()
    assert features['item_mean_rating'].notna().all()


def test_user_movie_features_use_only_ratings_before_each_event():
    ratings = pd.DataFrame(
        {
            'userId': [1, 1, 1],
            'movieId': [10, 10, 11],
            'rating': [4.0, 2.0, 5.0],
            'timestamp': [100, 200, 300],
        }
    )

    features = build_user_movie_features(ratings)

    assert features.loc[0, 'user_mean_rating'] == 3.0
    assert features.loc[0, 'user_rating_count'] == 0
    assert features.loc[1, 'user_mean_rating'] == 4.0
    assert features.loc[1, 'item_mean_rating'] == 4.0
    assert features.loc[1, 'user_item_interaction_count'] == 1


def test_temporal_split_features_are_safe_for_future_validation():
    ratings = pd.DataFrame(
        {
            'userId': [1, 1, 2, 2, 3, 3],
            'movieId': [10, 11, 10, 12, 11, 13],
            'rating': [4.5, 3.5, 2.5, 5.0, 4.0, 3.0],
            'timestamp': [100, 200, 300, 400, 500, 600],
        }
    )

    train_df, val_df = make_temporal_split_features(ratings, validation_fraction=0.33)

    assert train_df['timestamp'].max() < val_df['timestamp'].min()
    assert {'user_mean_rating', 'item_mean_rating'}.issubset(train_df.columns)
    assert {'user_mean_rating', 'item_mean_rating'}.issubset(val_df.columns)
