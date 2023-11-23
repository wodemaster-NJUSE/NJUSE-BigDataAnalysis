import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data

def preprocess_data_svm(data):
    df = pd.DataFrame(data)
    df = df.dropna()
    return df

def split_data_svm(df):
    X_train, X_test, y_train, y_test = train_test_split(df['headline'], df['category'], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def tfidf_transform_svm(X_train, X_test):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer
