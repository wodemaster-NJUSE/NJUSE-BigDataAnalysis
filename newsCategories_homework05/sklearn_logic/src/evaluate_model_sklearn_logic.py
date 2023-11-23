from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, X_test_tfidf, y_test):
    # 在测试集上进行预测
    y_pred = model.predict(X_test_tfidf)

    # 打印混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f'Confusion Matrix:\n{conf_matrix}')

    # 打印分类报告
    class_report = classification_report(y_test, y_pred)
    print(f'Classification Report:\n{class_report}')
