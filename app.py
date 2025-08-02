from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
import urllib.parse
import os

app = Flask(__name__)

# Load dataset and drop missing values
data = pd.read_csv('spotify_data.csv')
data.dropna(inplace=True)

# Add release year (assumes 'year' column exists)
data['release_year'] = data['year']

# Features to use
features = ['popularity', 'loudness', 'instrumentalness', 'tempo', 'valence', 'energy', 'danceability']

# Encode genre
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
genre_encoded = encoder.fit_transform(data[['genre']])

# Combine features
X_full = pd.concat([
    pd.DataFrame(genre_encoded, columns=encoder.get_feature_names_out()),
    data[features].reset_index(drop=True)
], axis=1)
X_full.dropna(inplace=True)
X_full.columns = X_full.columns.astype(str)

# Scale features
scaler_full = MinMaxScaler()
X_scaled_full = scaler_full.fit_transform(X_full)

scaler_numeric = MinMaxScaler()
X_numeric = data[features]
X_scaled_numeric = scaler_numeric.fit_transform(X_numeric)

# Fit base KNN models
knn_model_full = NearestNeighbors(n_neighbors=10, metric='euclidean')
knn_model_full.fit(X_scaled_full)

knn_model_numeric = NearestNeighbors(n_neighbors=10, metric='euclidean')
knn_model_numeric.fit(X_scaled_numeric)

# Filter by song type (latest/old/both)
def filter_by_song_type(song_type_choice, df):
    current_year = datetime.now().year
    if song_type_choice == "latest":
        return df[df['release_year'] >= current_year - 5]
    elif song_type_choice == "old":
        return df[df['release_year'] < current_year - 5]
    return df

# Recommend similar songs
def recommend_song(user_input, knn_model, scaler):
    user_scaled = scaler.transform([user_input])
    distances, indices = knn_model.kneighbors(user_scaled)
    return data.iloc[indices[0]][['artist_name', 'track_name', 'popularity', 'genre', 'release_year']]

# YouTube & Spotify links
def generate_youtube_search_url(query):
    return f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"

def generate_spotify_search_url(query):
    return f"https://open.spotify.com/search/{urllib.parse.quote(query)}"

@app.route('/')
def index():
    genres = data['genre'].dropna().unique()
    return render_template('index.html', genres=genres)

@app.route('/recommend', methods=['POST'])
def recommend():
    genre_choice = request.form['genre']
    popularity = float(request.form['popularity'])
    loudness = float(request.form['loudness'])
    instrumentalness = float(request.form['instrumentalness'])
    tempo = float(request.form['tempo'])
    valence = float(request.form['valence'])
    energy = float(request.form['energy'])
    danceability = float(request.form['danceability'])
    song_type = request.form['song_type']

    filtered_data = filter_by_song_type(song_type, data).dropna()

    if filtered_data.empty:
        return render_template('recommendations.html', recommendations="<p>No songs found for the selected criteria.</p>")

    if genre_choice != "all":
        filtered_data = filtered_data[filtered_data['genre'] == genre_choice].dropna()
        if filtered_data.empty:
            return render_template('recommendations.html', recommendations="<p>No songs found for the selected genre and criteria.</p>")

        genre_encoded_filtered = encoder.transform(filtered_data[['genre']])
        X_filtered_full = pd.concat([
            pd.DataFrame(genre_encoded_filtered, columns=encoder.get_feature_names_out()),
            filtered_data[features].reset_index(drop=True)
        ], axis=1)
        X_filtered_full.dropna(inplace=True)
        X_scaled_filtered_full = scaler_full.transform(X_filtered_full)

        knn_model_filtered = NearestNeighbors(n_neighbors=10, metric='euclidean')
        knn_model_filtered.fit(X_scaled_filtered_full)

        genre_input = encoder.transform([[genre_choice]])
        user_input = list(genre_input[0]) + [popularity, loudness, instrumentalness, tempo, valence, energy, danceability]

        recommendations = recommend_song(user_input, knn_model_filtered, scaler_full)
    else:
        user_input = [popularity, loudness, instrumentalness, tempo, valence, energy, danceability]
        recommendations = recommend_song(user_input, knn_model_numeric, scaler_numeric)

    if recommendations.empty:
        return render_template('recommendations.html', recommendations="<p>No songs found matching your preferences.</p>")

    recommendations['youtube_link'] = recommendations.apply(
        lambda row: generate_youtube_search_url(f"{row['artist_name']} {row['track_name']}"), axis=1
    )
    recommendations['spotify_link'] = recommendations.apply(
        lambda row: generate_spotify_search_url(f"{row['artist_name']} {row['track_name']}"), axis=1
    )

    recommendations_html = recommendations.to_html(index=False, escape=False, formatters={
        'youtube_link': lambda x: f'<a href="{x}" target="_blank">YouTube</a>',
        'spotify_link': lambda x: f'<a href="{x}" target="_blank">Spotify</a>'
    })

    return render_template('recommendations.html', recommendations=recommendations_html)

if __name__ == '__main__':
    app.run(debug=True)
