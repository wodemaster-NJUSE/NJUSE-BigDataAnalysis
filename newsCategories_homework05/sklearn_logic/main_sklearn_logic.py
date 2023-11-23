from src.preprocess_sklearn_logic import load_data, preprocess_data, split_data
from src.train_model_sklearn_logic import tfidf_transform, train_logistic_regression
from src.evaluate_model_sklearn_logic import evaluate_model

# 数据加载和预处理
data = load_data(r'data\News_Category.json')
df = preprocess_data(data)
X_train, X_test, y_train, y_test = split_data(df)

# 特征提取和模型训练
X_train_tfidf, X_test_tfidf, tfidf_vectorizer = tfidf_transform(X_train, X_test)
model = train_logistic_regression(X_train_tfidf, y_train)

# 模型评估
evaluate_model(model, X_test_tfidf, y_test)

