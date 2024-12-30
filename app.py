import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import gdown
import os

# Title and Description
st.title("Enhanced Movie Recommendation System")
st.markdown("""
Welcome to the Enhanced Movie Recommendation System! Enter your favorite movie, and we'll recommend similar movies with detailed information, including genres, cast, directors, ratings, and runtime.
""")

# URLs for IMDb datasets
DATASETS = {
    "title.principals.tsv.gz": "https://drive.google.com/uc?id=1ykVu-zhIY2jQCyPgMOiumQ1lgfpXvS05",
    "name.basics.tsv.gz": "https://drive.google.com/uc?id=1aY6nJJK-z1GAah-QROa77N19SsUpI-hF",
    "title.crew.tsv.gz": "https://drive.google.com/uc?id=1bpUAdyJMOuntT3Hn9_RGQIuhB8zLjPLu",
    "title.ratings.tsv.gz": "https://drive.google.com/uc?id=1rdWojWUUdLuPjrO7GCZbKUTMdkdjAOdz"
}

# Function to download datasets
@st.cache_data
def download_datasets():
    for filename, url in DATASETS.items():
        if not os.path.exists(filename):
            gdown.download(url, filename, quiet=False)

# Download the datasets
download_datasets()

# Load IMDb datasets
@st.cache_data
def load_data():
    basics = pd.read_csv("title.basics.tsv.gz", sep="\t", na_values="\\N", low_memory=False)
    crew = pd.read_csv("title.crew.tsv.gz", sep="\t", na_values="\\N", low_memory=False)
    principals = pd.read_csv("title.principals.tsv.gz", sep="\t", na_values="\\N", low_memory=False)
    ratings = pd.read_csv("title.ratings.tsv.gz", sep="\t", na_values="\\N", low_memory=False)
    return basics, crew, principals, ratings

basics, crew, principals, ratings = load_data()

# Filter for movies only
movies_data = basics[basics['titleType'] == 'movie']
movies_data = movies_data[['tconst', 'primaryTitle', 'genres', 'runtimeMinutes', 'startYear']]

# Merge with crew data to get directors
movies_data = pd.merge(movies_data, crew[['tconst', 'directors']], on='tconst', how='left')

# Merge with principals to get cast
cast_data = principals[principals['category'] == 'actor']
cast_data = cast_data.groupby('tconst')['nconst'].apply(lambda x: ', '.join(x)).reset_index()
cast_data.rename(columns={'nconst': 'cast'}, inplace=True)
movies_data = pd.merge(movies_data, cast_data, on='tconst', how='left')

# Merge with ratings to get average ratings and votes
movies_data = pd.merge(movies_data, ratings, on='tconst', how='left')

# Fill missing values
movies_data['genres'] = movies_data['genres'].fillna('')
movies_data['cast'] = movies_data['cast'].fillna('')
movies_data['directors'] = movies_data['directors'].fillna('')
movies_data['runtimeMinutes'] = movies_data['runtimeMinutes'].fillna('Unknown')
movies_data['averageRating'] = movies_data['averageRating'].fillna('No Rating')
movies_data['numVotes'] = movies_data['numVotes'].fillna(0)

# Combine features for recommendation
combined_features = movies_data['genres'] + ' ' + movies_data['cast'] + ' ' + movies_data['directors']

# Feature Vectorization
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Cosine Similarity
similarity = cosine_similarity(feature_vectors)

# User Input with Dropdown
list_of_all_titles = movies_data['primaryTitle'].tolist()
movie_name = st.selectbox("Enter your favorite movie name:", options=["Select a movie"] + list_of_all_titles)
if movie_name and movie_name != "Select a movie":
    # Get the index of the movie
    index_of_the_movie = movies_data[movies_data.primaryTitle == movie_name].index[0]

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
                "Title": movies_data.iloc[index]['primaryTitle'],
                "Genre": movies_data.iloc[index]['genres'],
                "Runtime (Minutes)": movies_data.iloc[index]['runtimeMinutes'],
                "Year": movies_data.iloc[index]['startYear'],
                "Director": movies_data.iloc[index]['directors'],
                "Cast": movies_data.iloc[index]['cast'],
                "Rating": movies_data.iloc[index]['averageRating'],
                "Votes": int(movies_data.iloc[index]['numVotes'])
            })
    
    recommended_df = pd.DataFrame(recommended_movies)
    st.table(recommended_df)
else:
    st.info("Please select a movie to get recommendations.")
