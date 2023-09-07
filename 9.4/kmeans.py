import os
import csv
import random
import numpy as np

def loadData(filename):
    data = open(filename, 'r', encoding="utf-8")
    reader = csv.reader(data)
    headers = next(reader)
    dataset = []
    for row in reader:
        row[1] = float(row[1])
        row[2] = float(row[2])
        dataset.append([row[1], row[2]])
    
    return np.array(dataset)


## 定义欧式距离
def euclidean_distance(x,y):
    """
    Input: vector x
    Output: vector y
    """
    # 初始化距离
    distance = 0
    # 遍历并对距离的平方进行累加
    for i in range(len(x)):
        distance += (x[i] - y[i]) ** 2
    return np.sqrt(distance)


## 质心初始化
def centroids_init(X,k):
    """
    Input:
    X: train example
    k: number of centroids
    Output:
    centroids: centroids maxtrix
    """
    m,n = X.shape
    centroids = np.zeros((k,n))
    for i in range(k):
        centroid = X[np.random.choice(range(m))]
        centroids[i] = centroid
    return centroids


## 定义样本所属最近的索引
def closest_centroid(x, centroids):
    """
    输入:
    x: 单个样本实例
    输出:
    closest_i
    """
    closest_i, min_distance = 0, np.inf
    for i, centroid in enumerate(centroids):
        distance = euclidean_distance(x, centroid)
        if distance < min_distance:
            min_distance = distance
            closest_i = i
    return closest_i

## 分配样本与构建簇
def build_cluster(centroids, k, X):
    """
    输入:
    centroids: 质心矩阵
    k: 质心个数, 也是聚类个数
    X: 训练样本
    输出：
    聚类簇
    """
    clusters = [[] for _ in range(k)]
    for x_i, x in enumerate(X):
        centroid_i = closest_centroid(x, centroids)
        clusters[centroid_i].append(x_i)
    return clusters


## 计算质心
def calculate_centroids(clusters, k, X):
    """
    输入：
    clusters: 上一步的聚类簇
    k: 质心个数, 也是聚类个数
    X: 训练样本, Numpy数组
    输出:
    centroids: 更新后的质心矩阵
    """
    # 特征数
    n = X.shape[1]
    # 初始化质心矩阵、大小为质心个数* 特征数
    centroids = np.zeros((k,n))
    # 遍历当前簇
    for i, cluster in enumerate(clusters):
        # 计算每个簇的均值作为新的质心
        centroid = np.mean(X[cluster], axis=0)
        print(centroid)
        centroids[i] = centroid
    return centroids


# 获取每个样本所属的聚类类别
def get_cluster_labels(clusters, X):
    """
    输入:
    clusters: 当前的聚类簇
    X: 训练样本
    输出:
    y_pred: 预测类别
    """
    # 预测结果初始化
    y_pred = np.zeros(X.shape[0])
    # 遍历聚类簇
    for cluster_i, cluster in enumerate(clusters):
        # 遍历当前簇
        for sample_i in cluster:
            y_pred[sample_i] = cluster_i
    return y_pred


### k均值聚类算法流程封装
def kmeans(X, k, epoches):
    """
    输入:
    X: 训练样本
    y: 质心个数, 也是聚类个数
    输出:
    预测类别列表
    """
    # 初始化质心
    centroids = centroids_init(X,k)
    # 遍历迭代求解:
    for _ in range(epoches):
        clusters =  build_cluster(centroids, k, X)
        cur_centroids = centroids
        centroids = calculate_centroids(clusters, k, X)
        diff = centroids - cur_centroids
        if not diff.any():
            break
    # 返回最终的聚类标签
    return get_cluster_labels(clusters, X)


if __name__ == "__main__":
    X = loadData("watermelon_data_4.0.csv")
    labels = kmeans(X, 2, epoches=10)
    print(labels)
    