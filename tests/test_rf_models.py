import pandas as pd

from src.validation.model_baselines import (
    build_random_forest_model,
    evaluate_model_on_split,
)


def test_random_forest_model_training_and_prediction():
    train_df = pd.DataFrame(
        {
            'userId': [1, 1, 2, 2, 3, 3],
            'movieId': [10, 11, 10, 12, 11, 13],
            'rating': [4.0, 5.0, 2.0, 3.0, 4.5, 5.0],
            'timestamp': [100, 200, 300, 400, 500, 600],
            'user_mean_rating': [4.5, 4.5, 2.5, 2.5, 4.75, 4.75],
            'user_rating_count': [2, 2, 2, 2, 2, 2],
            'item_mean_rating': [3.0, 4.0, 2.5, 3.0, 4.5, 5.0],
            'item_rating_count': [2, 2, 2, 2, 2, 2],
            'user_item_interaction_count': [1, 1, 1, 1, 1, 1],
        }
    )
    val_df = train_df.copy()

    model = build_random_forest_model(train_df, target='rating')
    prediction = model.predict(val_df.drop(columns='rating'))
    metrics = evaluate_model_on_split(val_df['rating'], prediction)

    assert prediction.shape[0] == len(val_df)
    assert {'rmse', 'mae', 'r2'}.issubset(metrics)
    assert metrics['rmse'] >= 0
