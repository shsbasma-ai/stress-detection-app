import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import joblib
from preprocess import clean_text
from preprocess import detect_language
from emotion_features import extract_emotion_features
import numpy as np 

# Charger et nettoyer les données
data = pd.read_csv("C:/Users/pc/Downloads/PFE2/stress_data.csv")
data["text"] = data["text"].apply(clean_text)
X = data["text"]
y = data["label"]      # (0-100)

# Vectorisation TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X).toarray()


# Deep + emotion features

extra_features = np.array([extract_emotion_features(t) for t in X])

# Fusion

X_final = np.hstack([X_tfidf, extra_features])
print("TFIDF shape :", X_tfidf.shape)
print("Extra shape :", extra_features.shape)
print("Final shape :", X_final.shape)

# Train

model = LinearRegression()
model.fit(X_final, y)

# Sauvegarder le modèle et le vectorizer
joblib.dump(model,"C:/Users/pc/Downloads/PFE2/model/stress_model.joblib")
joblib.dump(vectorizer, "C:/Users/pc/Downloads/PFE2/model/vectorizer.joblib")

print("\nModèle entraîné avec succès !")

# Niveau du stress
def stress_level(score):
    if score < 34:
        return "Low"
    elif score < 67:
        return "Medium"
    else:
        return "High"

# CHARGER LES SUGGESTIONS
suggestions_df = pd.read_csv("C:/Users/pc/Downloads/PFE2/suggestions.csv")

def get_suggestion(level, lang):
    # Filtrer level+lang
    filtered = suggestions_df[
        (suggestions_df["level"].str.lower() == level.lower()) &
        (suggestions_df["lang"] == lang)
    ]

    
    if filtered.empty:
        filtered = suggestions_df[
            (suggestions_df["level"].str.lower() == level.lower()) &
            (suggestions_df["lang"] == "mixed")
        ]

    
    if filtered.empty:
        filtered = suggestions_df[
            suggestions_df["level"].str.lower() == level.lower()
        ]

    return filtered.sample(1)["suggestion"].values[0]

# TEST PREDICTION 
def predict_text(text):

    clean_t = clean_text(text)

    # TFIDF
    tfidf_vec = vectorizer.transform([clean_t]).toarray()

    # Deep + emotion
    extra_vec = np.array([extract_emotion_features(clean_t)])

    # Fusion
    final_vec = np.hstack([tfidf_vec, extra_vec])

    score = model.predict(final_vec)[0]
    score = max(0, min(100, score))

    level = stress_level(score)
    lang = detect_language(text)

    suggestion = get_suggestion(level, lang)

    return score, level, lang, suggestion

##
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE :", mae)
print("RMSE :", rmse)

print(predict_text("Je suis très stressée et fatiguée ces jours-ci"))
print(predict_text("الحمد لله أنا مرتاحة اليوم"))
print(predict_text("I feel overwhelmed and anxious"))

