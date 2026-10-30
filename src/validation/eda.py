from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.loaders import load_movie_lens_data
from src.validation.data_quality import save_data_quality_report


def build_eda(raw_data_dir: str | Path, figures_dir: str | Path) -> dict:
    """Generate the initial EDA plots and summary metrics for the MovieLens data."""
    raw_dir = Path(raw_data_dir)
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    ratings_df, movies_df, users_df = load_movie_lens_data(raw_dir)

    ratings_df = ratings_df.copy()
    ratings_df["datetime"] = pd.to_datetime(ratings_df["timestamp"], unit="s")

    user_reviews = ratings_df.groupby("userId").size().sort_values(ascending=False)
    movie_reviews = ratings_df.groupby("movieId").size().sort_values(ascending=False)
    rating_distribution = ratings_df["rating"].value_counts().sort_index()

    # Genre distribution at the movie level
    genre_rows = movies_df["genres"].dropna().str.split("|")
    flattened_genres = [genre for row in genre_rows if isinstance(row, list) for genre in row]
    genre_counts = pd.Series(flattened_genres).value_counts()

    # Top popular titles
    title_popularity = (
        ratings_df.merge(movies_df[["movieId", "title"]], on="movieId", how="left")
        .groupby("title")
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

    # Top active users
    top_users = user_reviews.head(10)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.histplot(rating_distribution, bins=10, kde=False)
    plt.title("Rating distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figures_path / "ratings_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    top_users.plot(kind="bar")
    plt.title("Top users by number of ratings")
    plt.xlabel("UserId")
    plt.ylabel("Review count")
    plt.tight_layout()
    plt.savefig(figures_path / "top_users_by_reviews.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    movie_reviews.head(15).plot(kind="bar")
    plt.title("Top movies by number of ratings")
    plt.xlabel("MovieId")
    plt.ylabel("Review count")
    plt.tight_layout()
    plt.savefig(figures_path / "top_movies_by_reviews.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    genre_counts.head(15).plot(kind="bar")
    plt.title("Most common movie genres")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figures_path / "genre_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    ratings_df.set_index("datetime").resample("ME").size().plot(kind="line", marker="o")
    plt.title("Ratings over time")
    plt.xlabel("Month")
    plt.ylabel("Number of ratings")
    plt.tight_layout()
    plt.savefig(figures_path / "ratings_over_time.png", dpi=200)
    plt.close()

    summary = {
        'n_ratings': int(len(ratings_df)),
        'n_movies': int(ratings_df['movieId'].nunique()),
        'n_users': int(ratings_df['userId'].nunique()),
        'avg_ratings_per_user': float(ratings_df.groupby('userId').size().mean()),
        'avg_ratings_per_movie': float(ratings_df.groupby('movieId').size().mean()),
        'top_user_reviews': top_users.head(5).to_dict(),
        'top_movie_reviews': movie_reviews.head(5).to_dict(),
        'top_titles': title_popularity.to_dict(),
        'genre_counts': genre_counts.head(10).to_dict(),
    }

    save_data_quality_report(raw_dir, Path("reports/data_quality_report.md"))
    return summary


def generate_notebook(output_path: str | Path) -> None:
    """Write a notebook that documents the initial data understanding steps."""
    import json

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# MovieLens 1M - Data understanding\n",
                    "\n",
                    "This notebook loads the raw MovieLens files and explores the data before any modeling step.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "from src.data.loaders import load_movie_lens_data\n",
                    "\n",
                    "raw_dir = Path('../data/raw/ml-1m')\n",
                    "ratings_df, movies_df, users_df = load_movie_lens_data(raw_dir)\n",
                    "print(f'Ratings: {len(ratings_df)}')\n",
                    "print(f'Movies: {len(movies_df)}')\n",
                    "print(f'Users: {len(users_df)}')\n",
                    "ratings_df.head()\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "ratings_df['rating'].value_counts().sort_index().plot(kind='bar')\n",
                    "plt.title('Rating distribution')\n",
                    "plt.xlabel('Rating')\n",
                    "plt.ylabel('Count')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "user_counts = ratings_df.groupby('userId').size().sort_values(ascending=False)\n",
                    "movie_counts = ratings_df.groupby('movieId').size().sort_values(ascending=False)\n",
                    "user_counts.head(10)\n",
                    "movie_counts.head(10)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "genres = movies_df['genres'].str.split('|').explode()\n",
                    "genre_counts = genres.value_counts()\n",
                    "genre_counts.head(10).plot(kind='bar')\n",
                    "plt.title('Genre distribution')\n",
                    "plt.xlabel('Genre')\n",
                    "plt.ylabel('Count')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.9"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(notebook, handle)


if __name__ == "__main__":
    output_dir = Path("reports/figures")
    summary = build_eda("data/raw/ml-1m", output_dir)
    generate_notebook("notebooks/01_data_understanding.ipynb")
    print(summary)
    print("Initial EDA generated successfully.")
