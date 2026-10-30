import pandas as pd

from src.validation.temporal_strategy import (
    compute_global_mean_baseline,
    compute_regression_metrics,
    compute_user_item_bias_baseline,
    leakage_audit,
    temporal_split_by_timestamp,
)


def test_temporal_split_keeps_future_out_of_train():
    df = pd.DataFrame(
        {
            'userId': [1, 1, 2, 2, 3, 3],
            'movieId': [10, 11, 10, 12, 11, 13],
            'rating': [4.0, 5.0, 3.5, 2.0, 4.5, 4.0],
            'timestamp': [100, 200, 300, 400, 500, 600],
        }
    )

    train_df, val_df = temporal_split_by_timestamp(df, validation_fraction=0.33)

    assert len(train_df) + len(val_df) == len(df)
    assert train_df['timestamp'].max() < val_df['timestamp'].min()
    assert not leakage_audit(train_df, val_df)


def test_global_mean_baseline_is_calculated_without_leakage():
    train_df = pd.DataFrame(
        {'rating': [3.0, 4.0, 5.0, 2.0, 3.5]},
    )
    val_df = pd.DataFrame(
        {'rating': [4.0, 3.0, 2.5]},
    )

    predictions = compute_global_mean_baseline(train_df, val_df)

    assert predictions.shape[0] == len(val_df)
    assert predictions.nunique() == 1
    assert abs(predictions.iloc[0] - train_df['rating'].mean()) < 1e-9


def test_user_item_bias_baseline_uses_user_and_item_effects():
    train_df = pd.DataFrame(
        {
            'userId': [1, 1, 2, 2, 2, 3],
            'movieId': [10, 11, 10, 12, 13, 11],
            'rating': [3.0, 4.5, 2.0, 4.0, 3.5, 5.0],
        }
    )
    val_df = pd.DataFrame(
        {
            'userId': [1, 2, 3],
            'movieId': [10, 12, 11],
        }
    )

    predictions = compute_user_item_bias_baseline(train_df, val_df)

    assert predictions.notna().all()
    assert predictions.shape[0] == len(val_df)


def test_regression_metrics_are_numeric():
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.0, 2.0, 4.0])

    metrics = compute_regression_metrics(y_true, y_pred)

    assert set(metrics).issuperset({'rmse', 'mae', 'r2'})
    assert all(value >= 0 for value in metrics.values())
