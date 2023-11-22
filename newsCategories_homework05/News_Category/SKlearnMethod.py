# -*- coding: utf-8 -*-
from sklearn.datasets import News_Categoy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class MyLogicRegression():
    def __init__(self):
        self.news = News_Categoy()

    def run(self):
        x_train = self.news.data
        y_train = self.news.target
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                            test_size=0.2,
                                                            random_state=0,
                                                            stratify=y_train)
        # logitic 回归的分类模型
        ns = LogisticRegression()
        ns.fit(x_train, y_train)

        result = ns.predict(x_test)
        print('预测的结果', result)
        print('实际的结果', y_test)

if __name__ == '__main__':
    my_logic_regression = MyLogicRegression()
    my_logic_regression.run()