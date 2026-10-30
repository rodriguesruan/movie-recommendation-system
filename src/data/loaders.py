from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple

import pandas as pd


MOVIELENS_URL = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
MOVIELENS_FILES = ('ratings.dat', 'movies.dat', 'users.dat')


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


def ensure_movie_lens_data(base_dir: str | Path) -> Path:
    """Return the MovieLens directory, downloading required files if missing."""
    base_path = Path(base_dir)
    target_dir = base_path if base_path.name == 'ml-1m' else base_path / 'ml-1m'
    try:
        return resolve_movie_lens_dir(base_path)
    except FileNotFoundError:
        pass

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        archive_path = temporary_path / 'ml-1m.zip'
        try:
            urllib.request.urlretrieve(MOVIELENS_URL, archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                for filename in MOVIELENS_FILES:
                    member_name = f'ml-1m/{filename}'
                    with archive.open(member_name) as source:
                        with (temporary_path / filename).open('wb') as destination:
                            shutil.copyfileobj(source, destination)
        except (OSError, zipfile.BadZipFile, KeyError) as error:
            raise FileNotFoundError(
                f'Could not download a valid MovieLens dataset from {MOVIELENS_URL}.'
            ) from error

        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in MOVIELENS_FILES:
            shutil.copy2(temporary_path / filename, target_dir / filename)

    return resolve_movie_lens_dir(target_dir)


def _read_movielens_table(
    file_path: Path,
    names: list[str],
    dtype: dict[str, str],
) -> pd.DataFrame:
    """Normalize MovieLens's ``::`` delimiter and parse it with pandas' C engine."""
    with tempfile.NamedTemporaryFile(suffix='.tsv', delete=False) as converted_file:
        converted_path = Path(converted_file.name)
        with file_path.open('rb') as source:
            for line in source:
                converted_file.write(line.replace(b'::', b'\t'))

    try:
        return pd.read_csv(
            converted_path,
            sep='\t',
            engine='c',
            encoding='latin-1',
            names=names,
            dtype=dtype,
        )
    finally:
        converted_path.unlink(missing_ok=True)


def load_movie_lens_data(base_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the MovieLens 1M ratings, movies and users tables.

    Returns:
        ratings_df, movies_df, users_df
    """
    data_dir = resolve_movie_lens_dir(base_dir)

    ratings_df = _read_movielens_table(
        data_dir / 'ratings.dat',
        names=['userId', 'movieId', 'rating', 'timestamp'],
        dtype={'userId': 'int64', 'movieId': 'int64', 'rating': 'float64', 'timestamp': 'int64'},
    )

    movies_df = _read_movielens_table(
        data_dir / 'movies.dat',
        names=['movieId', 'title', 'genres'],
        dtype={'movieId': 'int64', 'title': 'string', 'genres': 'string'},
    )

    users_df = _read_movielens_table(
        data_dir / 'users.dat',
        names=['userId', 'gender', 'age', 'occupation', 'zip_code'],
        dtype={
            'userId': 'int64',
            'gender': 'string',
            'age': 'int64',
            'occupation': 'int64',
            'zip_code': 'string',
        },
    )

    return ratings_df, movies_df, users_df
