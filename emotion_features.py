import re
import numpy as np
from sentence_transformers import SentenceTransformer

# Charger modèle BERT 
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Lexiques émotionnels classiques
STRESS_WORDS = ["stress", "pressure", "ضغط", "mqal9"]
ANXIETY_WORDS = ["anxious", "worried", "قلق"]
FATIGUE_WORDS = ["tired", "exhausted", "عيان"]
NEGATIVE_WORDS = ["sad", "hate", "pain"]
POSITIVE_WORDS = ["calm", "relax", "happy", "bikhir"]


def count_words(text, words):
    return sum(1 for w in words if re.search(rf"\b{w}\b", text))


# Features émotionnelles + context
def manual_features(text):
    text = text.lower()

    length = len(text.split())
    exclam = text.count("!")
    question = text.count("?")

    stress = count_words(text, STRESS_WORDS)
    anxiety = count_words(text, ANXIETY_WORDS)
    fatigue = count_words(text, FATIGUE_WORDS)
    negative = count_words(text, NEGATIVE_WORDS)
    positive = count_words(text, POSITIVE_WORDS)

    return np.array([
        length,
        exclam,
        question,
        stress,
        anxiety,
        fatigue,
        negative,
        positive
    ])


# Embedding 
def deep_embedding(text):
    return bert_model.encode(text)  


# Fusion 
def extract_emotion_features(text):

    manual = manual_features(text)      
    deep = deep_embedding(text)         

    return np.hstack([manual, deep])   

