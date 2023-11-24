# -*- coding: utf-8 -*-
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from src.preprocess_svm import load_data
class MyTextSVM():
    def __init__(self):
        # 请根据你的实际数据路径进行修改
        self.data = load_data(r'data\News_Category.json')

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = [json.loads(line) for line in file]
        return data

    def preprocess_data(self, data):
        df = pd.DataFrame(data)
        # 处理缺失值
        df = df.dropna()

        return df

    def split_data(self, df):
        # 分割数据集为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(df['headline'], df['category'], test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def tfidf_transform(self, X_train, X_test):
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        return X_train_tfidf, X_test_tfidf

    def train_svm(self, X_train_tfidf, y_train, kernel='linear', C=1.0):
        svm_classifier = SVC(C=C, kernel=kernel, decision_function_shape='ovr')
        svm_classifier.fit(X_train_tfidf, y_train)
        return svm_classifier

    def run(self, kernel='linear', C=1.0):
        df = self.preprocess_data(self.data)
        X_train, X_test, y_train, y_test = self.split_data(df)
        X_train_tfidf, X_test_tfidf = self.tfidf_transform(X_train, X_test)

        svm_classifier = self.train_svm(X_train_tfidf, y_train, kernel, C)

        print("核函数:" + kernel + "，惩罚参数:" + str(C))
        print("训练集准确率:", svm_classifier.score(X_train_tfidf, y_train))
        print("测试集准确率:", svm_classifier.score(X_test_tfidf, y_test))

if __name__ == '__main__':
    my_text_svm = MyTextSVM()
    my_text_svm.run()
