
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model = None
vectorizer = None

def train():
    global model, vectorizer
    df = pd.read_csv("dataset.csv")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["Text"])
    y = df["Emotion"]
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, "emotion_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

def detect_emotion(text):
    global model, vectorizer
    if model is None:
        model = joblib.load("emotion_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
    return model.predict(vectorizer.transform([text]))[0]
