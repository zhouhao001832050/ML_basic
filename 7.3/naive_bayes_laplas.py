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
    # 每种特征的属性个数
    # numList = [] 
    # for i in range(len(features)-2):
    #     numList.append(len(featureDic[features[i]]))

    dataSet = np.array(dataSet)
    return dataSet, features

def cntProLap(dataSet, index, feature, value, classLabel, N):
    data_feat = dataSet[dataSet[:, -1]==classLabel]
    # prior_probability = len(data_feat) / len(dataSet)

    cnt = 0
    for data in data_feat:
        if data[index] == value:
            cnt += 1

    return cnt+1 / (len(data_feat) + N)

def naiveBayesClassifier(dataSet, features):
    dict = {}
    for index, feature in enumerate(features):
        dict[feature] = {}
        if feature != '密度' and feature != '含糖量':
            for value in featureDic[feature]:
                feature_value_len = len(featureDic[feature])
                D1 = cntProLap(dataSet, index, feature, value, '0', feature_value_len)
                D0 = cntProLap(dataSet, index, feature, value, "1", feature_value_len)
                dict[feature][value] = {}
                dict[feature][value]["是"] = D1
                dict[feature][value]["否"] = D0
        else:
            for label in ["1", "0"]:
                data_feat = dataSet[dataSet[:, -1]== label]
                extr = data_feat[:, index].astype("float64")
                aveg = extr.mean()
                var = extr.var()

                labelStr = ""
                if label == '1':
                    labelStr = '是'
                else:
                    labelStr = '否'

                dict[feature][labelStr] = {}
                dict[feature][labelStr]["平均值"] = aveg
                dict[feature][labelStr]["方差"] = var


    length = len(dataSet)
    classLabels = dataSet[:, -1].tolist()
    dict["好瓜"] = {}
    dict["好瓜"]['是'] = (classLabels.count('1') + 1) / (float(length) + 2)
    dict["好瓜"]['否'] = (classLabels.count('0') + 1) / (float(length) + 2)
    return dict

def NormDist(mean, var, x):
    return exp(-((float(x) - mean) ** 2) / (2 * var)) / (sqrt(2 * pi * var))


def predict(data, features, bayes_dict):
    pGood = bayes_dict["好瓜"]["是"]
    pBad = bayes_dict["好瓜"]["否"]
    # print(features)
    for index, feature in enumerate(features):
        if feature != "密度" and feature != "含糖量":
            print(feature)
            pGood *= bayes_dict[feature][data[index]]["是"]
            pBad *= bayes_dict[feature][data[index]]["否"]
        else:
            pGood *= NormDist(bayes_dict[feature]["是"]["平均值"], bayes_dict[feature]["是"]["方差"], data[index])

            pBad *= NormDist(bayes_dict[feature]["否"]["平均值"], bayes_dict[feature]["否"]["方差"], data[index])
    res = ""
    if pGood > pBad:
        res = "好瓜"
    else:
        res = "坏瓜"
    
    return pGood, pBad, res


if __name__ == "__main__":
    dataSet, features = getDataSet()
    dic = naiveBayesClassifier(dataSet, features)
    p1, p0, pre = predict(dataSet[0], features, dic)
    print(f"p1 = {p1}")
    print(f"p0 = {p0}")
    print(f"pre = {pre}")