Music Recommendation System

Overview:

This is a Flask-based Music Recommendation System that suggests songs based on user preferences such as genre, popularity, loudness, instrumentalness, tempo, valence, energy, and danceability. It uses the K-Nearest Neighbors (KNN) algorithm to find similar songs from a dataset.

Features:

Accepts user preferences for genre and song characteristics.

Filters recommendations based on the release year (latest/old songs).

Utilizes KNN for song recommendations.

Provides links to YouTube and Spotify for listening.

Web-based interface using Flask.

Tech Stack:

Python

Flask

Pandas

Scikit-learn

HTML/CSS (for frontend)

Dataset:

The dataset used for this project is spotify_data.csv, which contains various song features, including genre, popularity, tempo, and energy levels.
