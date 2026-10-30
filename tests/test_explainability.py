import pandas as pd

from src.validation.explainability import build_explanation_summary


def test_build_explanation_summary_returns_ordered_features():
    df = pd.DataFrame(
        {
            'user_mean_rating': [4.5, 3.0],
            'item_mean_rating': [4.0, 2.5],
            'user_rating_count': [100, 5],
            'item_rating_count': [50, 10],
            'prediction': [4.2, 2.9],
        }
    )

    explanation = build_explanation_summary(df)

    assert 'top_features' in explanation
    assert len(explanation['top_features']) >= 2
    assert explanation['top_features'][0]['feature'] in {'user_mean_rating', 'item_mean_rating', 'user_rating_count', 'item_rating_count'}
