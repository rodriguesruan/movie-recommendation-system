import pytest
import pandas as pd
from fastapi.testclient import TestClient

from src.api import create_app


@pytest.fixture
def client():
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
    return TestClient(create_app(ratings, movies))


def test_recommendations_endpoint_returns_unseen_ranked_movies(client):
    response = client.get('/recommendations/1?top_n=3')

    assert response.status_code == 200
    payload = response.json()
    assert payload['user_id'] == 1
    assert len(payload['recommendations']) <= 3
    assert payload['recommendations'][0]['movieId'] not in {10, 11}
    assert {'prediction', 'explanation'}.issubset(payload['recommendations'][0])


def test_recommendations_endpoint_rejects_unknown_user(client):
    response = client.get('/recommendations/999')

    assert response.status_code == 404


def test_recommendations_endpoint_validates_top_n(client):
    response = client.get('/recommendations/1?top_n=0')

    assert response.status_code == 422
