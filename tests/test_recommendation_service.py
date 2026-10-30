import pandas as pd

import src.validation.recommendation_service as recommendation_service
from src.validation.recommendation_service import generate_recommendations_for_user


class FixedScoreModel:
    feature_names_in_ = [
        'userId',
        'movieId',
        'user_mean_rating',
        'user_rating_count',
        'item_mean_rating',
        'item_rating_count',
        'user_item_interaction_count',
    ]

    def predict(self, features):
        return [5.0 if movie_id == 12 else 1.0 for movie_id in features['movieId']]


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


def test_recommendations_are_ranked_by_model_prediction():
    ratings = pd.DataFrame(
        {
            'userId': [1, 1, 2, 2],
            'movieId': [10, 11, 12, 13],
            'rating': [4.0, 4.0, 1.0, 5.0],
            'timestamp': [100, 200, 150, 300],
        }
    )
    movies = pd.DataFrame(
        {
            'movieId': [10, 11, 12, 13, 14],
            'title': ['A', 'B', 'C', 'D', 'E'],
            'genres': ['Drama', 'Comedy', 'Action', 'Drama', 'Sci-Fi'],
        }
    )

    results = generate_recommendations_for_user(
        ratings,
        movies,
        user_id=1,
        top_n=3,
        model=FixedScoreModel(),
    )

    assert results[0]['movieId'] == 12
    assert results[0]['prediction'] == 5.0
    assert 'historical average' in results[0]['explanation']


def test_training_samples_before_building_features(monkeypatch):
    ratings = pd.DataFrame(
        {
            'userId': [1, 1, 2, 2, 3, 3],
            'movieId': [10, 11, 12, 13, 14, 15],
            'rating': [4.0, 5.0, 3.0, 2.0, 4.5, 3.5],
            'timestamp': [100, 200, 300, 400, 500, 600],
        }
    )
    original_builder = recommendation_service.build_user_movie_features
    rows_seen_by_feature_builder = []

    def track_feature_input(sampled_ratings):
        rows_seen_by_feature_builder.append(len(sampled_ratings))
        return original_builder(sampled_ratings)

    monkeypatch.setattr(
        recommendation_service,
        'build_user_movie_features',
        track_feature_input,
    )

    model = recommendation_service.train_recommendation_model(ratings, max_training_rows=4)

    assert rows_seen_by_feature_builder == [4]
    assert len(model.estimators_) == 40
