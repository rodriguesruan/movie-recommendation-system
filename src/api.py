from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from src.data.loaders import load_movie_lens_data
from src.validation.recommendation_service import generate_recommendations_for_user


def create_app(
    ratings_df: pd.DataFrame | None = None,
    movies_df: pd.DataFrame | None = None,
) -> FastAPI:
    app = FastAPI(title='Movie Recommendation API', version='1.0.0')
    app.state.ratings_df = ratings_df
    app.state.movies_df = movies_df

    def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
        if app.state.ratings_df is None or app.state.movies_df is None:
            data_dir = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'ml-1m'
            try:
                ratings, movies, _ = load_movie_lens_data(data_dir)
            except FileNotFoundError as error:
                raise HTTPException(
                    status_code=503,
                    detail='MovieLens data is unavailable. Place the dataset in data/raw/ml-1m.',
                ) from error
            app.state.ratings_df = ratings
            app.state.movies_df = movies

        return app.state.ratings_df, app.state.movies_df

    @app.get('/health')
    def health() -> dict[str, str]:
        return {'status': 'ok'}

    @app.get('/recommendations/{user_id}')
    def recommendations(
        user_id: int,
        top_n: int = Query(default=10, ge=1, le=50),
    ) -> dict[str, object]:
        ratings, movies = get_data()
        if user_id not in set(ratings['userId'].unique()):
            raise HTTPException(status_code=404, detail=f'User {user_id} was not found.')

        results = generate_recommendations_for_user(ratings, movies, user_id=user_id, top_n=top_n)
        return {'user_id': user_id, 'recommendations': results}

    return app


app = create_app()