import pandas as pd

from src.validation.recommendation_service import generate_recommendations_for_user


def test_generate_recommendations_for_user_returns_ranked_results():
    ratings = pd.DataFrame(
        {
            'userId': [1, 1, 2, 2, 3],
            'movieId': [10, 11, 10, 12, 13],
            'rating': [4.0, 5.0, 3.0, 4.5, 4.0],
            'timestamp': [100, 200, 150, 300, 250],
        }
    )
    movies = pd.DataFrame(
        {
            'movieId': [10, 11, 12, 13, 14],
            'title': ['A', 'B', 'C', 'D', 'E'],
            'genres': ['Drama', 'Comedy', 'Action', 'Drama', 'Sci-Fi'],
        }
    )

    results = generate_recommendations_for_user(ratings, movies, user_id=1, top_n=3)

    assert isinstance(results, list)
    assert len(results) <= 3
    first = results[0]
    assert {'movieId', 'title', 'prediction', 'explanation'}.issubset(first.keys())
    assert first['movieId'] not in {10, 11}
