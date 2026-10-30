from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.data.loaders import load_movie_lens_data
from src.validation.recommendation_service import generate_recommendations_for_user


st.set_page_config(page_title='Movie Recommendations', page_icon='🎬', layout='wide')
st.title('Movie Recommendation Dashboard')

try:
    ratings_df, movies_df, users_df = load_movie_lens_data(Path('data/raw/ml-1m'))
except FileNotFoundError:
    st.warning('MovieLens data was not found. Download the dataset under data/raw/ml-1m before running the app.')
    st.stop()

user_ids = sorted(users_df['userId'].unique().tolist())
selected_user = st.selectbox('Select user', user_ids)
top_n = st.slider('Number of recommendations', min_value=1, max_value=10, value=5)

recommendations = generate_recommendations_for_user(
    ratings_df,
    movies_df,
    user_id=int(selected_user),
    top_n=top_n,
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
        use_container_width=True,
    )
