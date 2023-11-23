from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model_svm(model, X_test_tfidf, y_test):
    y_pred = model.predict(X_test_tfidf)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f'Confusion Matrix:\n{conf_matrix}')
    class_report = classification_report(y_test, y_pred)
    print(f'Classification Report:\n{class_report}')
