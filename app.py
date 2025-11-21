import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# ---------- Preprocessing Function ----------
def preprocess_comment(comment):
    comment = comment.lower().strip()
    comment = re.sub(r'\n', ' ', comment)
    comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

    stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    comment = ' '.join([word for word in comment.split() if word not in stop_words])

    lemmatizer = WordNetLemmatizer()
    comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

    return comment

# ---------- Load Model ----------
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("./lgbm_model.pkl")
    vectorizer = joblib.load("./tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# ---------- Streamlit UI ----------
st.title("📊 YouTube Comment Sentiment Analysis")
st.write("Analyze comment sentiment using Machine Learning (Positive, Neutral, Negative).")

# Text input
comment_text = st.text_area("✍ Enter your comment:", "")

if st.button("Predict Sentiment"):
    if comment_text:
        cleaned_text = preprocess_comment(comment_text)
        transformed = vectorizer.transform([cleaned_text]).toarray()
        prediction = int(model.predict(transformed)[0])

        label_map = {1: "Positive 😊", 0: "Neutral 😐", -1: "Negative 😠"}
        st.success(f"Sentiment: **{label_map[prediction]}**")
    else:
        st.warning("Please enter a comment first!")

# ---------- Word Cloud Section ----------
st.subheader("☁ Generate Word Cloud")
wordcloud_text = st.text_area("Enter text for Word Cloud:", "")

if st.button("Generate Word Cloud"):
    if wordcloud_text:
        wc = WordCloud(width=800, height=400, background_color='black').generate(wordcloud_text)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.warning("Please enter text to generate a Word Cloud!")

# ---------- Sentiment Pie Chart ----------
st.subheader("📈 Sentiment Pie Chart")
pos = st.number_input("Positive Count", min_value=0, value=5)
neu = st.number_input("Neutral Count", min_value=0, value=3)
neg = st.number_input("Negative Count", min_value=0, value=2)

if st.button("Generate Pie Chart"):
    fig, ax = plt.subplots()
    ax.pie([pos, neu, neg], labels=['Positive', 'Neutral', 'Negative'], autopct='%1.1f%%')
    ax.axis('equal')
    st.pyplot(fig)
