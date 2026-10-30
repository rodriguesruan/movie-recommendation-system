from __future__ import annotations

import pandas as pd

from src.validation.final_recommendations import build_recommendation_candidates, score_recommendations


def generate_recommendations_for_user(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    user_id: int,
    top_n: int = 10,
) -> list[dict]:
    """Create a ranked list of recommendations for a specific user.

    Returns a list of dictionaries suitable for JSON serialization or a Streamlit
    table, containing the movie metadata plus the predicted score and a human-
    readable explanation.
    """
    candidates = build_recommendation_candidates(ratings_df, movies_df, user_id=user_id, top_n=top_n)
    if candidates.empty:
        return []

    if {'user_mean_rating', 'item_mean_rating'} - set(candidates.columns):
        user_mean = ratings_df.groupby('userId')['rating'].mean()
        item_mean = ratings_df.groupby('movieId')['rating'].mean()

        candidates = candidates.copy()
        candidates['user_mean_rating'] = float(user_mean.get(user_id, ratings_df['rating'].mean()))
        candidates['item_mean_rating'] = candidates['movieId'].map(item_mean).fillna(ratings_df['rating'].mean())

    scored = score_recommendations(candidates)
    payload = scored[['movieId', 'title', 'genres', 'prediction', 'explanation']].copy()
    payload['score'] = payload['prediction']
    return payload.to_dict(orient='records')
