# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# import matplotlib
# from numpy import *

# matplotlib.rc("font", family='YouYuan')

# class LDA(object):
#     # 求出均值向量，类内三度矩阵和类间散度矩阵
#     def fit(self, X_, y_, plot_=False):
#         # 取出正反例各自数据，计算均值向量
#         neg = y_ == 0
#         pos = y_ == 1
#         X0 = X_[neg]
#         X1 = X_[pos]

#         # 均值向量，（1，2）
#         u0 = X0.mean(0, keepdims=True)
#         u1 = X1.mean(0, keepdims=True)

#         # 类内散度矩阵
#         sw = np.dot((X0 - u0).T, (X0 - u0)) + np.dot((X1 - u1).T, (X1 - u1))


#         # 类间散度矩阵
#         w = np.dot(np.linalg.inv(sw), (u0 - u1).T).reshape(1, -1)

#         # 绘图
#         if plot_:
#             fig, ax = plt.subplots()
#             ax.spines['right'].set_color('none')
#             ax.spines['top'].set_color('none')
#             ax.spines['left'].set_position(('data', 0))
#             ax.spines['bottom'].set_position(('data', 0))

#             # 画样本点
#             plt.scatter(X1[:, 0], X1[:, 1], c='k', marker='o', label='good')
#             plt.scatter(X0[:, 0], X0[:, 1], c='r', marker='x', label='bad')

#             plt.xlabel('密度')
#             plt.ylabel('含糖量')
#             plt.legend(loc='upper right')

#             # 画线
#             x_temp = np.linspace(-0.05, 0.15)
#             y_temp = x_temp * w[0, 1] / w[0, 0]
#             plt.plot(x_temp, y_temp, '#808080', linewidth=1)

#             wu = w / np.linalg.norm(w)

#             # 画正负样本点的投影，真的没看懂哈哈哈
#             X0_project = np.dot(X0, np.dot(wu.T, wu))
#             plt.scatter(X0_project[:, 0], X0_project[:, 1], c='r', s=15)
#             for i in range(X0.shape[0]):
#                 plt.plot([X0[i, 0], X0_project[i, 0]], [X0[i, 1], X0_project[i, 1]], '--r', linewidth=1)

#             X1_project = np.dot(X1, np.dot(wu.T, wu))
#             plt.scatter(X1_project[:, 0], X1_project[:, 1], c='k', s=15)
#             for i in range(X1.shape[0]):
#                 plt.plot([X1[i, 0], X1_project[i, 0]], [X1[i, 1], X1_project[i, 1]], '--r', linewidth=1)

#             # 均值向量的投影点
#             ax.annotate(r'u0 投影点',
#                         xy=(u0_project[:, 0], u0_project[:, 1]),
#                         xytext=(u0_project[:, 0] - 0.2, u0_project[:, 1] - 0.1),
#                         size=13,
#                         va="center", ha="left",
#                         arrowprops=dict(arrowstyle="->",
#                                         color="k",
#                                         )
#                         )

#             ax.annotate(r'u1 投影点',
#                         xy=(u1_project[:, 0], u1_project[:, 1]),
#                         xytext=(u1_project[:, 0] - 0.1, u1_project[:, 1] + 0.1),
#                         size=13,
#                         va="center", ha="left",
#                         arrowprops=dict(arrowstyle="->",
#                                         color="k",
#                                         )
#                         )
            
#             plt.axis("equal")  # 两坐标轴的单位刻度长度保存一致
#             plt.show()
            
#         self.w = w
#         self.u0 = u0
#         self.u1 = u1
#         return self


#     def predict(self, X):
#         # 各样本在的投影
#         project = np.dot(X, self.w.T)
#         # 均值投影
#         wu0 = np.dot(self.w, self.u0.T)
#         wu1 = np.dot(self.w, self.u1.T)

#         return (np.abs(project - wu1) < np.abs(project - wu0)).astype(int)


# if __name__ == "__main__":

#     X = data[:, 7:9].astype(float)
#     y = data[:, 9]
#     y = y.astype(int)
#     print(X)



########### LDA 算法流程 ############
"""
(1) 对训练集按类别进行分组;
(2) 分别计算每组样本的均值和协方差;
(3) 计算类间散度矩阵Sw;
(4) 计算两类样本的均值差U0-U1;
(5) 对类间散度矩阵Sw进行奇异值分解, 并求其逆;
(6) 根据Sw-1(U0-U1)得到w;
(7) 最后计算投影后的数据点Y=wX
"""

import numpy as np
from numpy import *

### 定义LDA类
class LDA:
    def __init__(self):
        # 初始化权重矩阵
        self.w=None
    
    # 协方差矩阵计算方法
    def calc_cov(self, X, Y=None):
        m = X.shape[0]
        # 数据标准化
        X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
        Y = X if Y == None else (Y - np.mean(Y, axis=0))/np.std(Y, axis=0)
        return 1 / m * np.matmul(X.T, Y)
    
    # LDA拟合方法
    def fit(self, X, y):
        # (1) 按类分组
        X0 = X[y==0]
        X1 = X[y==1]
        # (2) 分别计算两类数据自变量的协方差矩阵
        sigma0 = self.calc_cov(X0)
        sigma1 = self.calc_cov(X1)

        # (3) 计算类内散度矩阵
        Sw = sigma0 + sigma1

        # (4) 分别计算两类数据自变量的均值和差
        u0, u1 = np.mean(X0, axis=0), np.mean(X1, axis=0)

        mean_diff = np.atleast_1d(u0 - u1)

        # (5) 对类内散度矩阵进行奇异值分解
        U, S, V = np.linalg.svd(Sw)

        # (6) 计算类内散度矩阵的逆
        Sw_ = np.dot(np.dot(V.T, np.linalg.pinv(np.diag(S))), U.T)
        import pdb;pdb.set_trace()

        # 计算W
        self.w = Sw_.dot(mean_diff)

    
    # LDA分类预测
    def predict(self, X):
        # 初始化预测结果为空列表
        y_pred = []
        # 遍历待预测样本
        for x_i in X:
            # 模型预测
            h = x_i.dot(self.w)
            y = 1 * (h < 0)
            y_pred.append(y)
        return y_pred
            


# 数据测试
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 倒入iris数据集
# data = datasets.load_iris()
# print(data)
data = [[0.697,0.460,1],
        [0.774,0.376,1],
        [0.634,0.264,1],
        [0.608,0.318,1],
        [0.556,0.215,1],
        [0.403,0.237,1],
        [0.481,0.149,1],
        [0.437,0.211,1],
        [0.666,0.091,0],
        [0.243,0.267,0],
        [0.245,0.057,0],
        [0.343,0.099,0],
        [0.639,0.161,0],
        [0.657,0.198,0],
        [0.360,0.370,0],
        [0.593,0.042,0],
        [0.719,0.103,0]]
# 数据与标签
X = array([t[:2] for t in data])
y = array([t[-1] for t in data])
# print(X)
# print(y)
# 取标签不为2的数据
X_train,y_train = X[y!=2], y[y!=2]
# X_train = X_train.astype(float)
# y_train = y_train.astype(float)
# 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
# print(X_train)
# 创建LDA模型实例
lda = LDA()
lda.fit(X_train, y_train)
# LDA模型预测
y_pred = lda.predict(X_train)
# 测试集上的分类准确率
acc = accuracy_score(y_train, y_pred)
print("Accuracy of Numpy LDA:", acc)
