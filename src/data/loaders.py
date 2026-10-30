from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def resolve_movie_lens_dir(base_dir: str | Path) -> Path:
    """Resolve the project raw data directory to the actual MovieLens folder."""
    base_path = Path(base_dir)
    if base_path.is_dir() and {"ratings.dat", "movies.dat", "users.dat"}.issubset({p.name for p in base_path.iterdir()}):
        return base_path

    if base_path.is_dir():
        nested = base_path / "ml-1m"
        if nested.is_dir() and {"ratings.dat", "movies.dat", "users.dat"}.issubset({p.name for p in nested.iterdir()}):
            return nested

    raise FileNotFoundError(
        f"MovieLens raw data not found in '{base_path}'. Expected ratings.dat, movies.dat and users.dat."
    )


def load_movie_lens_data(base_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the MovieLens 1M ratings, movies and users tables.

    Returns:
        ratings_df, movies_df, users_df
    """
    data_dir = resolve_movie_lens_dir(base_dir)

    ratings_df = pd.read_csv(
        data_dir / "ratings.dat",
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["userId", "movieId", "rating", "timestamp"],
        dtype={"userId": "int64", "movieId": "int64", "rating": "float64", "timestamp": "int64"},
    )

    movies_df = pd.read_csv(
        data_dir / "movies.dat",
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["movieId", "title", "genres"],
        dtype={"movieId": "int64", "title": "string", "genres": "string"},
    )

    users_df = pd.read_csv(
        data_dir / "users.dat",
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["userId", "gender", "age", "occupation", "zip_code"],
        dtype={"userId": "int64", "gender": "string", "age": "int64", "occupation": "int64", "zip_code": "string"},
    )

    return ratings_df, movies_df, users_df
