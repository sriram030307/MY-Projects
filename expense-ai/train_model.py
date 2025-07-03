
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

model = None
vectorizer = None

def train():
    global model, vectorizer
    df = pd.read_csv("sample_statement.csv")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["Description"])
    y = df["Category"]
    model = MultinomialNB()
    model.fit(X, y)
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

def predict_category(desc):
    global model, vectorizer
    if model is None:
        model = joblib.load("model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
    return model.predict(vectorizer.transform([desc]))[0]
