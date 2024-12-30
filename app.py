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

# Show the first few rows of the dataset
st.subheader("Dataset Preview")
st.dataframe(movies_data.head())

# Data Preprocessing
st.subheader("Data Preprocessing")
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

# User Input
movie_name = st.text_input("Enter your favorite movie name:")
if movie_name:
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    if find_close_match:
        close_match = find_close_match[0]

        st.write(f"Close match found: **{close_match}**")

        # Get the index of the movie
        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

        # Get similar movies
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        # Display recommended movies
        st.subheader("Movies recommended for you:")
        recommended_movies = []
        for i, movie in enumerate(sorted_similar_movies):
            index = movie[0]
            title_from_index = movies_data[movies_data.index == index]['title'].values[0]
            if i < 30:
                recommended_movies.append(title_from_index)
        st.write(recommended_movies)
    else:
        st.error("No close match found. Please try a different movie name.")
