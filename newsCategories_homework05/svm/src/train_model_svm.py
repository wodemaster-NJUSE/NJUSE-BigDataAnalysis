from sklearn.svm import SVC

def train_model_svm(X_train_tfidf, y_train):
    model = SVC(kernel='linear')
    model.fit(X_train_tfidf, y_train)
    return model
