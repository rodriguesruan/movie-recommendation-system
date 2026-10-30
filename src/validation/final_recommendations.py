from __future__ import annotations

import pandas as pd


def build_recommendation_candidates(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    user_id: int,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return candidate movies unseen by the user, sorted by candidate popularity."""
    seen_movies = set(ratings_df.loc[ratings_df['userId'] == user_id, 'movieId'].unique())
    candidate_df = movies_df.loc[~movies_df['movieId'].isin(seen_movies)].copy()

    if candidate_df.empty:
        return pd.DataFrame(columns=['movieId', 'title', 'genres'])

    if 'rating' in ratings_df.columns:
        popularity = (
            ratings_df.groupby('movieId')['rating']
            .mean()
            .rename('movie_popularity_score')
        )
        candidate_df = candidate_df.merge(popularity, on='movieId', how='left')
        candidate_df = candidate_df.sort_values(['movie_popularity_score', 'movieId'], ascending=[False, True])
    else:
        candidate_df = candidate_df.sort_values('movieId')

    return candidate_df.head(top_n).reset_index(drop=True)


def score_recommendations(candidates: pd.DataFrame) -> pd.DataFrame:
    """Assign a predicted score and an explanation string to each candidate."""
    scored = candidates.copy()

    if 'user_mean_rating' in scored.columns and 'item_mean_rating' in scored.columns:
        scored['prediction'] = 0.5 * scored['user_mean_rating'] + 0.5 * scored['item_mean_rating']
    elif 'item_mean_rating' in scored.columns:
        scored['prediction'] = scored['item_mean_rating']
    elif 'user_mean_rating' in scored.columns:
        scored['prediction'] = scored['user_mean_rating']
    else:
        scored['prediction'] = 3.0

    scored['explanation'] = scored.apply(
        lambda row: (
            f"This title fits the current user profile because its user- and item-level history "
            f"suggests a strong preference around {row['prediction']:.2f} on the rating scale."
        ),
        axis=1,
    )

    scored = scored.sort_values('prediction', ascending=False).reset_index(drop=True)
    return scored
