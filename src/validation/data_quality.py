from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders import load_movie_lens_data


def build_data_quality_report(raw_data_dir: str | Path) -> dict:
    """Compute a compact quality report for the raw MovieLens files."""
    ratings_df, movies_df, users_df = load_movie_lens_data(raw_data_dir)

    rows = {
        'ratings_rows': int(len(ratings_df)),
        'movies_rows': int(len(movies_df)),
        'users_rows': int(len(users_df)),
    }

    report = {
        'rows': rows,
        'missing_values': {
            'ratings': int(ratings_df.isna().sum().sum()),
            'movies': int(movies_df.isna().sum().sum()),
            'users': int(users_df.isna().sum().sum()),
        },
        'duplicate_rows': {
            'ratings': int(ratings_df.duplicated().sum()),
            'movies': int(movies_df.duplicated().sum()),
            'users': int(users_df.duplicated().sum()),
        },
        'rating_range': {
            'min': float(ratings_df['rating'].min()),
            'max': float(ratings_df['rating'].max()),
        },
        'id_ranges': {
            'user_id': {'min': int(ratings_df['userId'].min()), 'max': int(ratings_df['userId'].max())},
            'movie_id': {'min': int(ratings_df['movieId'].min()), 'max': int(ratings_df['movieId'].max())},
        },
        'n_unique_users': int(ratings_df['userId'].nunique()),
        'n_unique_movies': int(ratings_df['movieId'].nunique()),
    }

    return report


def save_data_quality_report(raw_data_dir: str | Path, output_path: str | Path) -> dict:
    """Save the quality report in markdown format for the project reports folder."""
    report = build_data_quality_report(raw_data_dir)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        '# MovieLens 1M data quality report',
        '',
        '## Overview',
        '',
        f"- Ratings rows: {report['rows']['ratings_rows']}",
        f"- Movies rows: {report['rows']['movies_rows']}",
        f"- Users rows: {report['rows']['users_rows']}",
        f"- Unique users: {report['n_unique_users']}",
        f"- Unique movies: {report['n_unique_movies']}",
        '',
        '## Missing values',
        '',
        f"- Ratings: {report['missing_values']['ratings']}",
        f"- Movies: {report['missing_values']['movies']}",
        f"- Users: {report['missing_values']['users']}",
        '',
        '## Duplicates',
        '',
        f"- Ratings duplicates: {report['duplicate_rows']['ratings']}",
        f"- Movies duplicates: {report['duplicate_rows']['movies']}",
        f"- Users duplicates: {report['duplicate_rows']['users']}",
        '',
        '## Rating range',
        '',
        f"- Minimum rating: {report['rating_range']['min']}",
        f"- Maximum rating: {report['rating_range']['max']}",
        '',
        '## ID ranges',
        '',
        f"- UserId range: {report['id_ranges']['user_id']['min']} to {report['id_ranges']['user_id']['max']}",
        f"- MovieId range: {report['id_ranges']['movie_id']['min']} to {report['id_ranges']['movie_id']['max']}",
    ]

    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
