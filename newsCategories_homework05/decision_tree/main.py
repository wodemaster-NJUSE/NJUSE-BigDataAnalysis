# -*- coding: utf-8 -*-
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# 读取新闻分类数据集
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data

def preprocess_data(data):
    df = pd.DataFrame(data)
    df = df.dropna()  # 处理缺失值
    return df

# 分割数据集为训练集和测试集
def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(df['headline'], df['category'], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# 使用 TF-IDF 进行文本特征提取
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_transform(X_train, X_test):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer

# 训练决策树模型
def train_decision_tree(X_train_tfidf, y_train, max_depth, min_samples_split):
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    clf.fit(X_train_tfidf, y_train)
    return clf

# 评估模型
def evaluate_model(model, X_test_tfidf, y_test):
    y_pred = model.predict(X_test_tfidf)

    # 打印混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f'Confusion Matrix:\n{conf_matrix}')

    # 打印分类报告
    class_report = classification_report(y_test, y_pred)
    print(f'Classification Report:\n{class_report}')

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # 可视化决策树
    plt.figure(figsize=(12, 12))
    tree.plot_tree(model, filled=True, feature_names=None, class_names=None)
    plt.savefig("Decision_Tree.png", bbox_inches="tight", pad_inches=0.0)

if __name__ == '__main__':
    # 加载数据
    data = load_data("your_news_data.json")  # 请替换为你的新闻分类数据文件路径
    df = preprocess_data(data)

    # 分割数据
    X_train, X_test, y_train, y_test = split_data(df)

    # 特征提取
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = tfidf_transform(X_train, X_test)

    # 训练决策树模型
    decision_tree_model = train_decision_tree(X_train_tfidf, y_train, max_depth=4, min_samples_split=2)

    # 评估模型
    evaluate_model(decision_tree_model, X_test_tfidf, y_test)



# # main.py 这是另一个模型，使用到src中的代码
# from src.preprocess import load_data, preprocess_data, split_data
# from src.train_model import tfidf_transform, train_decision_tree
# from src.evaluate_model import evaluate_model

# # 数据加载和预处理
# data = load_data(r'data/News_Category.json')
# df = preprocess_data(data)
# X_train, X_test, y_train, y_test = split_data(df)

# # 特征提取和模型训练
# X_train_tfidf, X_test_tfidf, tfidf_vectorizer = tfidf_transform(X_train, X_test)
# model = train_decision_tree(X_train_tfidf, y_train)

# # 模型评估
# evaluate_model(model, X_test_tfidf, y_test)
