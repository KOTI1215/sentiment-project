from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('model.pkl')
vec = joblib.load('vec.pkl')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle HTML form submission
    if request.form:
        text = request.form.get('review', '')
    # Handle JSON API requests
    elif request.is_json:
        data = request.get_json()
        text = data.get('review', '')
    else:
        text = ''

    if not text:
        return render_template('index.html', error="Please enter a review.")

    # Model prediction
    text_vec = vec.transform([text])
    prediction = model.predict(text_vec)[0]
    probabilities = model.predict_proba(text_vec)[0]
    confidence = round(max(probabilities) * 100, 1)

    # Return HTML page with results for browser forms
    if request.form:
        return render_template('index.html', sentiment=prediction.upper(), confidence=confidence, review=text)
    
    # Return JSON for API calls
    return jsonify({'sentiment': prediction, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)