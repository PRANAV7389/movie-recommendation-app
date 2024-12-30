import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# Title and Description
st.title("Movie Recommendation System")
st.markdown("""
Welcome to the Movie Recommendation System! Enter your favorite movie, and we'll recommend similar movies based on genres, keywords, taglines, cast, and director.
""")

# Preloaded dataset
movies_data = pd.read_csv("movies.csv")

# Data Preprocessing
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + \
                    movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Feature Vectorization
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Cosine Similarity
similarity = cosine_similarity(feature_vectors)

# User Input with Dropdown
list_of_all_titles = movies_data['title'].tolist()
movie_name = st.selectbox("Enter your favorite movie name:", options=["Select a movie"] + list_of_all_titles)
if movie_name and movie_name != "Select a movie":
    # Get the index of the movie
    index_of_the_movie = movies_data[movies_data.title == movie_name]['index'].values[0]

    # Get similar movies
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # Display recommended movies in a table
    st.subheader("Movies recommended for you:")
    recommended_movies = []
    for i, movie in enumerate(sorted_similar_movies):
        index = movie[0]
        if i < 30:
            recommended_movies.append({
                "Title": movies_data.iloc[index]['title'],
                "Genre": movies_data.iloc[index]['genres'],
                "Tagline": movies_data.iloc[index]['tagline'],
                "Cast": movies_data.iloc[index]['cast'],
                "Director": movies_data.iloc[index]['director']
            })
    
    recommended_df = pd.DataFrame(recommended_movies)
    st.table(recommended_df)
else:
    st.info("Please select a movie to get recommendations.")
