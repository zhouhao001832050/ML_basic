"""
AdaBoost实现
参数T表示集成学习包含几个基学习器，也就是几个决策树。首先获取数据，然后进入for循环，循环T次，每一次将训练数据，训练标签，权重包含进D中，然后进行训练，得到学习器ht，然后计算一下ht的错误率et。这里主要要避免et=0的情况，防止一会除0。如果et>0.5则停止训练。然后将学习器ht存入学习器列表h_list，然后计算该学习器的权重alpha[t]，表示该学习器最终的"话语权"。

然后就是adaboost的重点，更新权重，adaboost通过 “降低预测正确样本的权重，提高预测错误样本的权重” 来保证这些基学习器的多样性，更新权重的公式如下：
Dt(x)表示数据x在第t轮训练的权重（对应代码中的变量w，w[i]表示第i条数据的权重，注意没有下标t，因为每次训练直接覆盖上次的权重了）。
Zt就是一个规范化因子，用于保证权重加和等于1（对应代码中的变量Z）。

"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets._samples_generator import make_blobs
from sklearn.metrics import accuracy_score


def get_data():
    x = []
    y = np.ndarray(shape=(17,))
    data = np.ndarray(shape=(17,2))
    # print(data)
    with open("watermelon_3.0.txt", 'r') as f:
        for i, line in enumerate(f.readlines()):
            d = {}
            d["密度"], d["含糖量"] = line.split(",")[0], line.split(",")[1]
            d["好瓜"] = line.split(",")[2].strip()
            # print(len(d["好瓜"]))
            # print(d["好瓜"][1])

            if d["好瓜"] == "是":
                d["好瓜"] = 1
            else:
                d["好瓜"] = 0
            data[i] = [d["密度"], d["含糖量"]]
            y[i] = d["好瓜"]
    # df_res = pd.DataFrame(data)
    # label = df_res["好瓜"]
    return data,y

class DecisionStump:
    def __init__(self):
        # 基于划分阈值决定样本分类为1或-1
        self.label = 1
        # 特征索引
        self.feature_index =None
        # 特征划分阈值
        self.threshold = None
        # 指示分类准确率的值
        self.alpha = None


# 定义Adaboost类
class Adaboost:
    # 定义弱分类器个数
    def __init__(self, n_estimators=5):
        self.n_estimators = n_estimators
    
    
    def calError(self, w, pred, y):
        err = 0
        for i, p in enumerate(pred):
            y_i = y[i]
            if p!= y_i:
                err += w[i]
        return err

    # Adaboost拟合算法
    def fit(self, X, y):
        m,n = X.shape
        # (1)初始化权重分布为均匀分布1/N
        w = np.full(m, (1/m))
        # 初始化基分类器列表
        self.estimators = []
        # (2)遍历n个弱分类器
        for n_est in range(self.n_estimators):
            # 训练一个弱分类器，决策树桩
            estimator = DecisionStump()
            # 设定一个最小化误差率
            min_error = float('inf')
            # print(f"第{n_est}个分类器")
            # 遍历数据集特征，根据最小分类误差率选择最小特征
            for i in range(n):
                # 获取特征值
                values = np.expand_dims(X[:,i], axis=1)
                # import pdb;pdb.set_trace()
                # 特征值去重
                unique_values = np.unique(values)
                # print(unique_values)
                # 将每一个特征值作为分类阈值
                # print(f"第{i}个特征")
                for threshold in unique_values:
                    p = 1
                    # 初始化所有预测值都是1
                    pred = np.ones(np.shape(y))
                    # print(np.shape(y))
                    pred[X[:,i] < threshold] = -1
                    # 计算误差率
                    # import pdb;pdb.set_trace()
                    # error = sum(w[y!=pred])
                    error = self.calError(w, pred, y)
                    # if error<0:
                    #     # import pdb;pdb.set_trace()
                    #     print(y)
                    #     print(pred)
                    #     print(w)
                    #     import pdb;pdb.set_trace()


                    # 根据分类误差进行翻转预测结果，待实验
                    #TODO
                    # if error > 0.5:
                    #     error = 1 - error
                    #     p = -1

                    # print(f"error的值:{error}")
                    if error < min_error and error > 0:
                        estimator.label=p
                        estimator.threshold=threshold
                        estimator.feature_index=i
                        min_error=error

            # 计算基分类器的权重
            estimator.alpha = 0.5 * np.log((1.0-min_error) / (min_error + 1e-9))

            # print(f"最小误差:{min_error}")
            # print(f"第{n_est}个分类器,alpha的值:{estimator.alpha}")
            # 初始化所有预测值为1
            preds = np.ones(np.shape(y))
            # 获取所有小于阈值的负类索引
            negative_idx = (estimator.label * X[:, estimator.feature_index] < estimator.label * estimator.threshold)
            # 将负类设为‘-1’
            preds[negative_idx] = -1
            # 更新样本权重
            w *= -estimator.alpha * y * preds
            # import pdb;pdb.set_trace()
            w /= np.sum(w)
            # 保存该弱分类器
            self.estimators.append(estimator)

    def predict(self, X):
        m = len(X)
        y_pred = np.zeros((m,1))
        # 计算每个弱分类器的预测值
        for estimator in self.estimators:
            # 初始化所有预测值为1
            predictions = np.ones(np.shape(y_pred))
            # 获取所有小于阈值的负类索引
            negative_idx = (estimator.label * X[:, estimator.feature_index] < estimator.label * estimator.threshold)
            # 将负类设为‘1’
            predictions[negative_idx] = -1
            # 对每个弱分类器的预测结果进行加权
            y_pred += estimator.alpha * predictions
        # 返回最终预测结果
        # import pdb;pdb.set_trace()
        y_pred = np.sign(y_pred).flatten()
        return y_pred

if __name__ == "__main__":
    # T = 5
    X,y = get_data()
    # X, y = make_blobs(n_samples=150,
    #                   n_features=2, 
    #                   centers=2, 
    #                   cluster_std=1.2, 
    #                   random_state=40)
    # print(type(X))
    # print(X.shape)
    # print(type(y))
    # print(y.shape)
    # print(X)
    # print(y)
    y_ = y.copy()
    y_[y_==0]=-1
    y_ = y_.astype(float)
    # X_train, X_test, y_train, y_test = train_test_split(X,y_,test_size=0.3, random_state=43)
    clf = Adaboost(n_estimators=5)
    clf.fit(X,y_)
    # 模型预测
    y_pred = clf.predict(X)
    # 计算模型的准确率
    accuracy = accuracy_score(y_, y_pred)
    print("Accuracy of Adaboost by numpy:", accuracy)



