import gradio as gr
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# 1️⃣ Load your trained sentiment model & tokenizer
# -----------------------------
sentiment_model = load_model("sentiment_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

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
# 4️⃣ Gradio interface
# -----------------------------
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter your text here..."),
    outputs=gr.Textbox(label="Sentiment"),
    title="Sentiment Analysis App",
    description="Type a sentence and get its sentiment (neutral / positive / negative) with confidence."
)

# -----------------------------
# 5️⃣ Launch app (local & public)
# -----------------------------
iface.launch(share=True)
