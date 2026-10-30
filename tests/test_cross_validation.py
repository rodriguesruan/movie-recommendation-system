import pandas as pd

from src.validation.cross_validation import (
    make_temporal_folds,
    optimize_random_forest,
)


def test_make_temporal_folds_builds_valid_splits():
    df = pd.DataFrame(
        {
            'userId': [1, 1, 2, 2, 3, 3, 4, 4],
            'movieId': [10, 11, 12, 13, 14, 15, 16, 17],
            'rating': [4.0, 3.5, 2.0, 5.0, 3.0, 4.0, 4.5, 2.5],
            'timestamp': [100, 200, 250, 350, 450, 550, 650, 750],
            'user_mean_rating': [4.0, 4.0, 2.0, 2.0, 3.0, 3.0, 4.5, 4.5],
            'item_mean_rating': [3.8, 3.8, 2.0, 2.0, 3.5, 3.5, 4.0, 4.0],
        }
    )

    folds = make_temporal_folds(df, n_splits=2)

    assert len(folds) == 2
    assert folds[0][0]['timestamp'].max() < folds[0][1]['timestamp'].min()
    assert folds[1][0]['timestamp'].max() < folds[1][1]['timestamp'].min()


def test_optimize_random_forest_returns_summary_dict():
    df = pd.DataFrame(
        {
            'userId': [1, 1, 2, 2, 3, 3],
            'movieId': [10, 11, 10, 12, 11, 13],
            'rating': [5.0, 4.0, 2.5, 3.5, 4.0, 5.0],
            'timestamp': [100, 200, 300, 400, 500, 600],
            'user_mean_rating': [4.5, 4.5, 3.0, 3.0, 4.5, 4.5],
            'item_mean_rating': [4.5, 4.5, 2.5, 3.5, 4.5, 5.0],
        }
    )

    result = optimize_random_forest(df, n_trials=2)

    assert 'best_params' in result
    assert 'cv_rmse' in result
    assert 'history' in result
