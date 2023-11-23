from src.preprocess_bpnn import load_data, preprocess_data_bpnn, split_data_bpnn, tfidf_transform_bpnn
from src.train_model_bpnn import train_model_bpnn
from src.evaluate_model_bpnn import evaluate_model_bpnn

# 数据加载和预处理
data = load_data(r'data/News_Category.json')
df = preprocess_data_bpnn(data)
X_train, X_test, y_train, y_test = split_data_bpnn(df)

# 特征提取和模型训练
X_train_tfidf, X_test_tfidf, tfidf_vectorizer = tfidf_transform_bpnn(X_train, X_test)
model = train_model_bpnn(X_train_tfidf, y_train)

# 模型评估
evaluate_model_bpnn(model, X_test_tfidf, y_test)
