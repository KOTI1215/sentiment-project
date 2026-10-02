import joblib
import re

model = joblib.load('model.pkl')
vectorizer = joblib.load('vec.pkl')

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

print("Sentiment Predictor (type 'quit' to exit)")
while True:
    review = input("\nReview: ")
    if review.lower() == 'quit':
        break
    cleaned = clean(review)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    confidence = max(prob) * 100
    print(f"Sentiment: {pred} ({confidence:.1f}%)")