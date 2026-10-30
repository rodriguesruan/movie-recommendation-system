import pandas as pd

from src.validation.final_recommendations import (
    build_recommendation_candidates,
    score_recommendations,
)


def test_build_recommendation_candidates_removes_seen_items():
    ratings = pd.DataFrame(
        {
            'userId': [1, 1, 2],
            'movieId': [10, 11, 10],
            'rating': [4.0, 5.0, 3.0],
            'timestamp': [100, 200, 150],
        }
    )
    movies = pd.DataFrame(
        {
            'movieId': [10, 11, 12, 13],
            'title': ['A', 'B', 'C', 'D'],
            'genres': ['Drama', 'Comedy', 'Action', 'Drama'],
        }
    )

    candidates = build_recommendation_candidates(ratings, movies, user_id=1, top_n=2)

    assert set(candidates['movieId']).issubset({12, 13})
    assert len(candidates) <= 2


def test_score_recommendations_adds_prediction_and_explanation():
    candidates = pd.DataFrame(
        {
            'movieId': [12, 13],
            'title': ['C', 'D'],
            'genres': ['Action', 'Drama'],
            'user_mean_rating': [4.2, 3.9],
            'item_mean_rating': [3.8, 4.1],
        }
    )

    scored = score_recommendations(candidates)

    assert {'prediction', 'explanation'}.issubset(scored.columns)
    assert scored['prediction'].notna().all()
    assert scored['explanation'].str.len().gt(0).all()
