# src/train_model.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

def tfidf_transform(X_train, X_test):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer

def train_decision_tree(X_train_tfidf, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train_tfidf, y_train)
    return model
