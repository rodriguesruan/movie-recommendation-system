from pathlib import Path
import tempfile
import zipfile

import pandas as pd

from src.data import loaders
from src.data.loaders import ensure_movie_lens_data, load_movie_lens_data


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


def test_ensure_movie_lens_data_downloads_only_required_files(monkeypatch):
    project_dir = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(dir=project_dir) as temporary_directory:
        temporary_path = Path(temporary_directory)
        archive_path = temporary_path / 'fixture.zip'

        with zipfile.ZipFile(archive_path, 'w') as archive:
            for filename in loaders.MOVIELENS_FILES:
                archive.writestr(f'ml-1m/{filename}', f'fixture-{filename}')

        def fake_download(url, destination):
            assert url == loaders.MOVIELENS_URL
            Path(destination).write_bytes(archive_path.read_bytes())

        monkeypatch.setattr(loaders.urllib.request, 'urlretrieve', fake_download)
        data_dir = ensure_movie_lens_data(temporary_path / 'data' / 'raw' / 'ml-1m')

        assert data_dir == temporary_path / 'data' / 'raw' / 'ml-1m'
        assert {path.name for path in data_dir.iterdir()} == set(loaders.MOVIELENS_FILES)
