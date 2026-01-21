import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# 1️⃣ Load your trained sentiment model & tokenizer
# -----------------------------
@st.cache_resource
def load_sentiment_model():
    model = load_model("sentiment_model.h5")
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    return model, tokenizer

sentiment_model, tokenizer = load_sentiment_model()

# -----------------------------
# 2️⃣ Mapping numeric outputs
# -----------------------------
sentiment_map = {0: "neutral", 1: "positive", 2: "negative"}

# -----------------------------
# 3️⃣ Prediction function
# -----------------------------
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    seq_pad = pad_sequences(seq, maxlen=18, padding='post')
    pred = sentiment_model.predict(seq_pad, verbose=0)
    class_idx = np.argmax(pred)
    confidence = pred[0][class_idx]
    return f"{sentiment_map[class_idx]} ({confidence*100:.1f}%)"

# -----------------------------
# 4️⃣ Streamlit app UI
# -----------------------------
st.title("Sentiment Analysis App (Streamlit)")
st.write("""
Type a sentence below and get its sentiment (neutral / positive / negative) with confidence.
""")

user_input = st.text_area("Enter your text here:")

if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        result = predict_sentiment(user_input)
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter some text to analyze.")
