import os




import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import operator

# 特征字典，后面用到了好多次，干脆当全局变量了
featureDic = {
    '色泽': ['浅白', '青绿', '乌黑'],
    '根蒂': ['硬挺', '蜷缩', '稍蜷'],
    '敲声': ['沉闷', '浊响', '清脆'],
    '纹理': ['清晰', '模糊', '稍糊'],
    '脐部': ['凹陷', '平坦', '稍凹'],
    '触感': ['硬滑', '软粘']}


def getDataSet():
    """
    get watermelon data set 3.0.
    :return: 编码好的数据集以及特征的字典。
    """
    dataSet = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, 1],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, 1],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, 1],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, 1],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, 1],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, 1],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, 1],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, 1],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, 0],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, 0],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, 0],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, 0],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, 0],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, 0],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, 0],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, 0],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, 0]
    ]

    features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖量']
    # features = ['color', 'root', 'knocks', 'texture', 'navel', 'touch', 'density', 'sugar']

    # #得到特征值字典，本来用这个生成的特征字典，还是直接当全局变量方便
    # featureDic = {}
    # for i in range(len(features)):
    #     featureList = [example[i] for example in dataSet]
    #     uniqueFeature = list(set(featureList))
    #     featureDic[features[i]] = uniqueFeature

    # 每种特征的属性个数
    # numList = []  # [3, 3, 3, 3, 3, 2]
    # for i in range(len(features) - 2):
    #     numList.append(len(featureDic[features[i]]))

    dataSet = np.array(dataSet)
    return dataSet, features


def AODE(dataSet, data, features):
    # 尝试将每个属性作为超父来构建SPODE
    m,n = dataSet.shape
    # n = n-3
    pDir = {}
    for classLabel in ["好瓜","坏瓜"]:
        P = 0.0
        if classLabel == "好瓜":
            sign = "1"
        else:
            sign = "0"
        extraData = dataSet[dataSet[:, -1]==sign]
        for i, feature in enumerate(features[:-2]):
            xi = data[i]
            Dcxi = extraData[extraData[:, i]==xi] # 第i个属性上取值为xi的样本数
            Ni = len(featureDic[feature])
            Pcxi = (len(Dcxi)+1)/ float(2*Ni+m)
            multi_j = 1
            for j, feature_j in enumerate(features[:-2]):
                xj = data[j]
                Dcxij= Dcxi[Dcxi[:, j]==xj]
                Nj=len(featureDic[feature_j])
                PCond = (len(Dcxij)+ 1) / float(len(Dcxi) + Nj)
                multi_j *= PCond
            P += Pcxi * multi_j
        pDir[classLabel] = P
    if pDir["好瓜"] > pDir["坏瓜"]:
        preClass = "好瓜"
    else:
        preClass = "坏瓜"
    return pDir, preClass


def main():
    dataSet, features = getDataSet()
    pDir, ans = AODE(dataSet, dataSet[0], features)
    print(f"prob = {pDir}")
    print(f"predict = {ans}")

if __name__ == "__main__":
    main()