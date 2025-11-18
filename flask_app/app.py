# app.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ---------- Preprocessing Function ----------
def preprocess_comment(comment):
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment


# ---------- Load Model (MLflow → Local Fallback) ----------
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    mlflow.set_tracking_uri("http://ec2-54-196-109-131.compute-1.amazonaws.com:5000/")

    vectorizer = joblib.load(vectorizer_path)
    model = None

    try:
        print("🔍 Trying to load model from MLflow Registry...")
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        print("✅ Model successfully loaded from MLflow!")
    except Exception as e:
        print(f"⚠️ MLflow load failed: {e}")
        print("👉 Loading model locally from .pkl file...")
        model = joblib.load("./lgbm_model.pkl")

    return model, vectorizer


# ---------- Initialize Model & Vectorizer ----------
model, vectorizer = load_model_and_vectorizer("my_model", "1", "./tfidf_vectorizer.pkl")


@app.route('/')
def home():
    return "Welcome to our Flask API (YouTube Sentiment Analysis)"


# ---------- Basic Sentiment Prediction ----------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        preprocessed = [preprocess_comment(c) for c in comments]
        transformed = vectorizer.transform(preprocessed)
        preds = model.predict(transformed).tolist()
        preds = [str(p) for p in preds]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [{"comment": c, "sentiment": s} for c, s in zip(comments, preds)]
    return jsonify(response)


# ---------- Prediction with Timestamp ----------
@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]
        preprocessed = [preprocess_comment(c) for c in comments]
        transformed = vectorizer.transform(preprocessed)
        preds = model.predict(transformed).tolist()
        preds = [str(p) for p in preds]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [{"comment": c, "sentiment": s, "timestamp": t} for c, s, t in zip(comments, preds, timestamps)]
    return jsonify(response)


# ---------- Generate Pie Chart ----------
@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')

        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


# ---------- Generate Word Cloud ----------
@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        preprocessed = [preprocess_comment(c) for c in comments]
        text = ' '.join(preprocessed)

        wordcloud = WordCloud(
            width=800, height=400, background_color='black',
            colormap='Blues', stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


# ---------- Generate Trend Graph ----------
@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)

        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
        totals = monthly_counts.sum(axis=1)
        percentages = (monthly_counts.T / totals).T * 100

        for val in [-1, 0, 1]:
            if val not in percentages.columns:
                percentages[val] = 0

        percentages = percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))
        colors = {-1: 'red', 0: 'gray', 1: 'green'}
        for val in [-1, 0, 1]:
            plt.plot(percentages.index, percentages[val], marker='o', linestyle='-', label={-1:'Negative',0:'Neutral',1:'Positive'}[val], color=colors[val])

        plt.title('Monthly Sentiment Trend Over Time')
        plt.xlabel('Month')
        plt.ylabel('Sentiment %')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


# ---------- Run App ----------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
