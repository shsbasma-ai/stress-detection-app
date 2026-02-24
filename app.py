from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import random

from preprocess import clean_text, detect_language
from emotion_features import extract_emotion_features

app = Flask(__name__)

# Load model
model = joblib.load("C:\\Users\\pc\\Downloads\\PFE2\\model\\stress_model.joblib")
vectorizer = joblib.load("C:\\Users\\pc\\Downloads\\PFE2\\model\\vectorizer.joblib")

# Load suggestions
suggestions_df = pd.read_csv("C:/Users/pc/Downloads/PFE2/suggestions.csv")

def stress_level(score):
    if score < 34:
        return "low"
    elif score < 67:
        return "medium"
    else:
        return "high"

def get_suggestion(level, lang):
    subset = suggestions_df[
        (suggestions_df["level"] == level) &
        (suggestions_df["lang"] == lang)
    ]
    if subset.empty:
        subset = suggestions_df[suggestions_df["level"] == level]
    return random.choice(subset["suggestion"].tolist())

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data["text"]

    clean_t = clean_text(text)
    tfidf_vec = vectorizer.transform([clean_t]).toarray()
    emotion_vec = np.array([extract_emotion_features(clean_t)])

    final_vec = np.hstack([tfidf_vec, emotion_vec])

    score = float(np.clip(model.predict(final_vec)[0], 0, 100))
    level = stress_level(score)
    lang = detect_language(text).lower()
    suggestion = get_suggestion(level, lang)

    return jsonify({
        "score": score,
        "level": level,
        "lang": lang,
        "suggestion": suggestion
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)