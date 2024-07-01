import streamlit as st
import pandas as pd
import pickle
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
movies = pd.read_csv('processed_data.csv')


cv = CountVectorizer(max_features=5000,stop_words='english')




vectors = cv.fit_transform(movies['tags']).toarray()



similarities = cosine_similarity(vectors)
def fetch_poster(movie_id):
    url ='https://api.themoviedb.org/3/movie/{}?api_key=c9c18c600b0eb79cf9b1770c68253a7c'.format(movie_id)
    data = requests.get(url).json()
    image = 'http://image.tmdb.org/t/p/w500/' + data['poster_path']
    return image

def recommend(movie_name):
    index = movies[movies['title'] == movie_name].index.item()
    recommended_movies_index = sorted(list(enumerate(similarities[index])),reverse=True,key=lambda x: x[1])[1:6]
    recommended_movies = []
    recommended_movies_posters = []
    for i in recommended_movies_index:
        recommended_movies.append(movies.iloc[i[0]]['title'])
        recommended_movies_posters.append(fetch_poster(movies.iloc[i[0]]['movie_id']))
    return recommended_movies, recommended_movies_posters


st.title('Movie Recommender System')
selected_movie = st.selectbox("Select a Movie",options=movies.title)
cols = st.columns(5)
similar_movies,movie_posters = recommend(selected_movie)
for idx,col in enumerate(cols):
    with col:
        st.text(similar_movies[idx])
        st.image(movie_posters[idx])


