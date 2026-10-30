import pandas as pd

from src.validation.ensemble import (
    build_ensemble_predictions,
    train_blending_model,
)


def test_build_ensemble_predictions_returns_column_for_each_model():
    ratings = pd.DataFrame(
        {
            'userId': [1, 2, 3],
            'movieId': [10, 11, 12],
            'rating': [4.0, 3.5, 5.0],
            'timestamp': [100, 200, 300],
            'user_mean_rating': [4.0, 3.5, 5.0],
            'item_mean_rating': [4.0, 3.5, 5.0],
        }
    )

    predictions = build_ensemble_predictions(ratings)

    assert {'global_mean', 'user_bias', 'item_bias', 'ensemble'}.issubset(predictions.columns)
    assert predictions['ensemble'].notna().all()


def test_train_blending_model_returns_valid_linear_weights():
    df = pd.DataFrame(
        {
            'rating': [3.0, 4.0, 5.0, 2.0],
            'global_mean': [3.0, 3.0, 3.0, 3.0],
            'user_bias': [2.8, 3.8, 4.8, 2.8],
            'item_bias': [3.1, 3.9, 4.9, 2.1],
        }
    )

    model = train_blending_model(df)

    assert model is not None
    assert list(model.keys()) == ['weights', 'intercept']
