# 🎬 Movie Review Sentiment Analysis Application

A full-stack Natural Language Processing (NLP) web application that predicts movie review sentiment (Positive/Negative) with real-time confidence scores, batch CSV processing, and query history logging.

🚀 **Live Web App:** [https://sentiment-project-3t3x.onrender.com](https://sentiment-project-3t3x.onrender.com)

---

## 📌 Features

- **Real-Time Sentiment Prediction:** Instant sentiment analysis with confidence percentage scoring.
- **Batch CSV Processing:** Upload a CSV file containing reviews to process in bulk and download structured predictions.
- **Prediction History Log:** Embedded SQLite database to track and render recent predictions on the dashboard.
- **Dual Support:** Serves both interactive HTML web forms and JSON API endpoints.
- **Interactive UI:** Pre-loaded test samples and dynamic result styling.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, SQLite3
- **Machine Learning & NLP:** Scikit-Learn (TF-IDF Bigrams + Logistic Regression), Pandas, Joblib
- **Frontend:** HTML5, CSS3, Jinja2
- **Deployment & Tooling:** Render, Git, GitHub, Gunicorn

---

## 🏃 Local Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/KOTI1215/sentiment-project.git](https://github.com/KOTI1215/sentiment-project.git)
   cd sentiment-project