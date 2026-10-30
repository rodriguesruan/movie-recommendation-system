from pathlib import Path

import pandas as pd

from src.data.loaders import load_movie_lens_data


DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'ml-1m'


def test_load_movie_lens_data_returns_expected_frames():
    ratings, movies, users = load_movie_lens_data(DATA_DIR)

    assert isinstance(ratings, pd.DataFrame)
    assert isinstance(movies, pd.DataFrame)
    assert isinstance(users, pd.DataFrame)

    assert not ratings.empty
    assert not movies.empty
    assert not users.empty

    assert {'userId', 'movieId', 'rating', 'timestamp'}.issubset(ratings.columns)
    assert {'movieId', 'title', 'genres'}.issubset(movies.columns)
    assert {'userId', 'gender', 'age', 'occupation', 'zip_code'}.issubset(users.columns)

    assert ratings['rating'].between(0.5, 5).all()
    assert ratings['userId'].notna().all()
    assert ratings['movieId'].notna().all()
    assert not ratings.duplicated(subset=['userId', 'movieId', 'timestamp']).any()
