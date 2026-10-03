import http
from unittest import result

from flask import Flask, redirect, render_template, request, jsonify, send_file
import joblib
import pandas as pd
import io
import csv
import sqlite3
import datetime
import requests


OMDB_API_KEY = "bfcae30c"

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('model.pkl')
vec = joblib.load('vec.pkl')

# Initialize SQLite Database
def init_db():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review TEXT,
            sentiment TEXT,
            confidence REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()
@app.route("/")
def home():
    init_db()
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    
    pos_count = 0
    neg_count = 0
    history = []

    try:
        cursor.execute("SELECT review, sentiment, confidence FROM predictions ORDER BY id DESC LIMIT 5")
        history = cursor.fetchall()

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE sentiment = 'POSITIVE'")
        pos_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE sentiment = 'NEGATIVE'")
        neg_count = cursor.fetchone()[0]
    except Exception:
        history = []
    finally:
        conn.close()

    return render_template("index.html", history=history, pos_count=pos_count, neg_count=neg_count)
def log_prediction(review, sentiment, confidence):
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO predictions (review, sentiment, confidence, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (review, sentiment, confidence, timestamp))
    conn.commit()
    conn.close()

def get_recent_history(limit=5):
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT review, sentiment, confidence, timestamp FROM predictions ORDER BY id DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

@app.route('/', methods=['GET'])
def index():
    history = get_recent_history()
    return render_template('index.html', history=history)

@app.route('/predict', methods=['POST'])
def predict():
    history = get_recent_history()
    
    # Handle single text input (Form)
    if request.form and 'review' in request.form:
        text = request.form.get('review', '')
        if not text.strip():
            return render_template('index.html', error="Please enter a review.", history=history)
        
        text_vec = vec.transform([text])
        prediction = model.predict(text_vec)[0]
        probabilities = model.predict_proba(text_vec)[0]
        confidence = round(max(probabilities) * 100, 1)
        
        log_prediction(text, prediction.upper(), confidence)
        history = get_recent_history()  # Refresh history after new entry
        
        return render_template('index.html', sentiment=prediction.upper(), confidence=confidence, review=text, history=history)

    # Handle JSON API requests
    elif request.is_json:
        data = request.get_json()
        text = data.get('review', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        text_vec = vec.transform([text])
        prediction = model.predict(text_vec)[0]
        probabilities = model.predict_proba(text_vec)[0]
        confidence = round(max(probabilities) * 100, 1)
        
        log_prediction(text, prediction.upper(), confidence)
        
        return jsonify({'sentiment': prediction, 'confidence': confidence})

    return render_template('index.html', error="Invalid request.", history=history)

@app.route('/predict-csv', methods=['POST'])
def predict_csv():
    history = get_recent_history()
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded.", history=history)
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected.", history=history)

    try:
        df = pd.read_csv(file)
        
        review_col = None
        for col in ['review', 'text', 'Review', 'Text']:
            if col in df.columns:
                review_col = col
                break
        
        if not review_col:
            return render_template('index.html', error="CSV must contain a column named 'review' or 'text'.", history=history)

        reviews = df[review_col].astype(str).tolist()
        text_vec = vec.transform(reviews)
        predictions = model.predict(text_vec)
        probabilities = model.predict_proba(text_vec)
        
        df['Sentiment'] = [p.upper() for p in predictions]
        df['Confidence (%)'] = [round(max(prob) * 100, 1) for prob in probabilities]

        output = io.BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='sentiment_results.csv'
        )
    except Exception as e:
        return render_template('index.html', error=f"Error processing CSV: {str(e)}", history=history)

@app.route("/get-movie-info", methods=["POST"])
def get_movie_info():
    data = request.get_json()
    movie_name = data.get("movie_name", "").strip()
    movie_year = data.get("movie_year", "").strip()

    if not movie_name:
        return jsonify({"error": "Please enter a movie title"}), 400

    url = f"https://www.omdbapi.com/?t={movie_name}&y={movie_year}&plot=full&apikey={OMDB_API_KEY}"
    response = requests.get(url)
    movie_data = response.json()

    if movie_data.get("Response") == "False":
        return jsonify({"error": "Movie not found!"}), 404

    result = {
        "title": movie_data.get("Title"),
        "year": movie_data.get("Year"),
        "rated": movie_data.get("Rated"),
        "released": movie_data.get("Released"),
        "genre": movie_data.get("Genre"),
        "director": movie_data.get("Director"),
        "actors": movie_data.get("Actors"),
        "plot": movie_data.get("Plot"),
        "poster": movie_data.get("Poster"),
        "imdb_rating": movie_data.get("imdbRating"),
        "box_office": movie_data.get("BoxOffice", "N/A"),
    }

    return jsonify(result)

@app.route("/export-history", methods=["GET"])
def export_history():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT review, sentiment, confidence, timestamp FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    # Generate CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Review", "Sentiment", "Confidence (%)", "Timestamp"])
    
    for row in rows:
        writer.writerow(row)

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="prediction_history.csv"
    )
@app.route("/clear-history", methods=["POST"])
def clear_history():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    return redirect("/")
if __name__ == "__main__":
    app.run(debug=True)