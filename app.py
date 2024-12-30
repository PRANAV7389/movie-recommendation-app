import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# Apply Custom CSS for Styling
def apply_custom_css():
    st.markdown(
        """
        <style>
        .main {background-color: #f0f0f0;}
        .stButton>button {background-color: #007BFF; color: white; font-size: 16px;}
        .stDownloadButton>button {background-color: #28a745; color: white; font-size: 16px;}
        .stSidebar {background-color: #343a40; color: white;}
        .stSidebar .sidebar-content {color: white;}
        h1 {color: #007BFF;}
        h2, h3, h4 {color: #343a40;}
        </style>
        """,
        unsafe_allow_html=True
    )

apply_custom_css()

# Title and Description
st.title("🎥 Enhanced Movie Recommendation System")
st.markdown(
    """
    Welcome to the **Enhanced Movie Recommendation System**! Enter your favorite movie, and we'll recommend similar movies with detailed insights.
    """
)

# Preloaded dataset
movies_data = pd.read_csv("movies.csv")

# Data Preprocessing
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + \
                    movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Feature Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
feature_vectors = vectorizer.fit_transform(combined_features)

# Cosine Similarity
similarity = cosine_similarity(feature_vectors)

# User Input with Dropdown
list_of_all_titles = movies_data['title'].tolist()
movie_name = st.text_input("🔍 Search for your favorite movie:")
if movie_name:
    close_matches = difflib.get_close_matches(movie_name, list_of_all_titles, n=5)
    if close_matches:
        movie_name = st.selectbox("Did you mean:", close_matches)
    else:
        st.error("⚠️ No close match found. Please try another movie.")

if movie_name and movie_name in list_of_all_titles:
    # Get the index of the movie
    index_of_the_movie = movies_data[movies_data.title == movie_name]['index'].values[0]

    # Get similar movies
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # Filter Recommendations
    st.sidebar.markdown("### Filters")
    min_vote_count = st.sidebar.slider("Minimum Votes:", 0, 1000, 100)
    min_rating = st.sidebar.slider("Minimum Rating:", 0.0, 10.0, 7.0)

    # Display recommended movies in an interactive table
    st.subheader("🎬 Movies Recommended for You:")
    recommended_movies = []
    for i, movie in enumerate(sorted_similar_movies):
        index = movie[0]
        movie_details = movies_data.iloc[index]
        if i < 30 and movie_details['vote_count'] >= min_vote_count and movie_details['vote_average'] >= min_rating:
            recommended_movies.append({
                "Title": movie_details['title'],
                "Genre": movie_details['genres'],
                "Tagline": movie_details['tagline'],
                "Cast": movie_details['cast'],
                "Director": movie_details['director'],
                "Release Date": movie_details['release_date'],
                "Rating": movie_details['vote_average'],
                "Votes": movie_details['vote_count'],
                "Revenue": f"${movie_details['revenue']:,}",
                "Budget": f"${movie_details['budget']:,}"
            })

    if recommended_movies:
        recommended_df = pd.DataFrame(recommended_movies)
        st.dataframe(recommended_df, use_container_width=True)
        # Allow users to download recommendations
        st.download_button("📥 Download Recommendations", recommended_df.to_csv(index=False), "recommendations.csv")
    else:
        st.warning("⚠️ No movies found matching your criteria.")

    # More Like This Feature
    if st.button("🔄 More Like This"):
        similar_index = sorted_similar_movies[1][0]
        similar_movie = movies_data.iloc[similar_index]['title']
        st.write(f"🎥 More movies like **{similar_movie}**")
else:
    st.info("ℹ️ Please enter a movie to get recommendations.")

# Data Insights Section
st.sidebar.title("📊 Insights")
st.sidebar.markdown("Explore insights from the dataset.")

if st.sidebar.checkbox("📈 Show Genre Distribution"):
    genre_counts = movies_data['genres'].value_counts().head(10)
    st.bar_chart(genre_counts)

if st.sidebar.checkbox("⭐ Top Rated Movies"):
    top_rated = movies_data.sort_values(by='vote_average', ascending=False).head(10)
    st.write(top_rated[['title', 'vote_average', 'genres']])

if st.sidebar.checkbox("🔥 Most Popular Movies"):
    most_popular = movies_data.sort_values(by='popularity', ascending=False).head(10)
    st.write(most_popular[['title', 'popularity', 'genres']])
