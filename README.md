🎧 Music Recommendation System
A smart, genre-aware music recommender built using Flask + Machine Learning that delivers personalized song suggestions based on mood, tempo, and energy levels.

📌 Overview
This project is a lightweight yet powerful Flask-based web app that recommends songs to users based on their musical preferences. By leveraging the K-Nearest Neighbors (KNN) algorithm and a curated Spotify dataset, it suggests tracks that match user-selected characteristics like danceability, loudness, energy, valence, and more.

🚀 Key Features
🎯 Custom Recommendations: Input your favorite music traits — genre, tempo, valence, and more — to get personalized results.

📅 Filter by Era: Choose between latest, old, or all songs.

🤖 ML-powered Suggestions: Uses a KNN model trained on scaled and encoded song data.

🔗 Quick Links to Listen: One-click access to songs on YouTube and Spotify.

🌐 Clean Web UI: Responsive, accessible interface built with HTML and CSS.

🛠 Tech Stack
Layer	Technology
Backend	Python, Flask
Machine Learning	scikit-learn, pandas, NumPy
Frontend	HTML5, CSS3 (no JS frameworks)
Model	K-Nearest Neighbors (KNN)
Dataset	Spotify tracks metadata with audio features

🧪 Sample Inputs
Feature	Example
Genre	dance, pop, rock, etc.
Popularity	0–100
Loudness (dB)	-60 to 0
Instrumentalness	0–100
Tempo (BPM)	50–250
Valence	0–100 (happiness level)
Energy	0–100
Danceability	0–100
Song Type	Latest, Old, or Both

📁 Dataset Used
spotify_data.csv contains:

🎵 Song Name & Artist

📊 Popularity metrics

🔊 Audio features: tempo, energy, valence, danceability, loudness, etc.

🧬 Genre labels

📆 Release year