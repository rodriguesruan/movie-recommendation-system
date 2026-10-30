from __future__ import annotations

import streamlit as st

from src.data.loaders import load_movie_lens_data
from src.data.loaders import ensure_movie_lens_data
from src.validation.recommendation_service import (
    generate_recommendations_for_user,
    train_recommendation_model,
)


st.set_page_config(page_title='Movie Recommendations', page_icon='🎬', layout='wide')
st.title('Movie Recommendation Dashboard')


@st.cache_resource(show_spinner='Loading MovieLens data and training the recommendation model...')
def load_recommendation_resources():
    data_dir = ensure_movie_lens_data('data/raw/ml-1m')
    ratings, movies, users = load_movie_lens_data(data_dir)
    model = train_recommendation_model(ratings)
    return ratings, movies, users, model


try:
    ratings_df, movies_df, users_df, recommendation_model = load_recommendation_resources()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(f'Could not initialize recommendations: {error}')
    st.stop()

user_ids = sorted(users_df['userId'].unique().tolist())
selected_user = st.selectbox('Select user', user_ids)
top_n = st.slider('Number of recommendations', min_value=1, max_value=10, value=5)

recommendations = generate_recommendations_for_user(
    ratings_df,
    movies_df,
    user_id=int(selected_user),
    top_n=top_n,
    model=recommendation_model,
)

if not recommendations:
    st.info('This user has already watched every available movie in the dataset.')
else:
    st.subheader(f'Top {len(recommendations)} recommendations for user {selected_user}')
    st.dataframe(
        [
            {
                'movieId': item['movieId'],
                'title': item['title'],
                'genres': item['genres'],
                'score': round(float(item['prediction']), 2),
                'explanation': item['explanation'],
            }
            for item in recommendations
        ],
        width='stretch',
    )
