from flask import Flask, render_template, request, jsonify, send_file
import joblib
import pandas as pd
import io
import sqlite3
import datetime

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

if __name__ == '__main__':
    app.run(debug=True)