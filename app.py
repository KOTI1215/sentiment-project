from flask import Flask, request, jsonify, render_template
import joblib
import re

app = Flask(__name__)

model = joblib.load('model.pkl')
vectorizer = joblib.load('vec.pkl')

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = clean(data['review'])
    vec = vectorizer.transform([review])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    confidence = float(max(prob)) * 100
    return jsonify({'sentiment': pred, 'confidence': round(confidence, 1)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)