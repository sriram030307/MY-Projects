
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def match_jobs(resume_text):
    df = pd.read_csv("job_dataset.csv")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['description'].tolist() + [resume_text])
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    df['match_score'] = cosine_sim[0]
    return df.sort_values(by='match_score', ascending=False)[['title', 'match_score']].head(5)
