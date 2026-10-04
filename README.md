# Full-Stack Movie Review Sentiment Analysis Web Application

A production-ready, full-stack machine learning web application that classifies movie reviews into **Positive** or **Negative** sentiments, integrates movie metadata via the OMDb API, provides batch CSV processing, and logs prediction history in an SQLite database.

## 🚀 Live Demo
* **Live Application:** [https://sentiment-project-3t3x.onrender.com](https://sentiment-project-3t3x.onrender.com)

## 🛠️ Tech Stack
* **Backend:** Python, Flask, Scikit-Learn
* **Database:** SQLite
* **Frontend:** HTML5, CSS3, Bootstrap, Chart.js
* **External APIs:** OMDb API (Open Movie Database)
* **Version Control & Hosting:** Git, GitHub, Render (CI/CD)

## ✨ Key Features
* **Single-Text Prediction:** Real-time sentiment analysis with confidence percentage scoring.
* **Batch CSV Processing:** Upload a CSV of reviews and instantly download predicted results.
* **Database Logging:** Stores all prediction history locally via SQLite.
* **Interactive Analytics:** Visualizes sentiment breakdown using Chart.js.
* **Movie Metadata Lookup:** Fetches posters, plots, and ratings dynamically using the OMDb API.
* **Data Management:** Export history to CSV or clear logs with a single click.

## 📦 Project Structure
sentiment-project/
│
├── app.py              # Main Flask application & routes
├── model.pkl           # Trained Scikit-Learn ML model
├── vec.pkl             # TF-IDF Vectorizer
├── history.db          # SQLite database for prediction logs
├── requirements.txt    # Python dependencies
├── Procfile            # Render deployment configuration
└── templates/
    └── index.html      # Frontend user interface