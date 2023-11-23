from src.preprocess_svm import load_data, preprocess_data_svm, split_data_svm, tfidf_transform_svm
from src.train_model_svm import train_model_svm
from src.evaluate_model_svm import evaluate_model_svm

# 数据加载和预处理
data = load_data(r'data/News_Category.json')
df = preprocess_data_svm(data)
X_train, X_test, y_train, y_test = split_data_svm(df)

# 特征提取和模型训练
X_train_tfidf, X_test_tfidf, tfidf_vectorizer = tfidf_transform_svm(X_train, X_test)

# 训练 SVM 模型
model = train_model_svm(X_train_tfidf, y_train)

# 模型评估
evaluate_model_svm(model, X_test_tfidf, y_test)
