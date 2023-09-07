import pandas as pd
import numpy as np
from math import log
import operator
import copy
import re
import sys


def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    return max(classCount)


#  根节点的信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts={}
    for entry in dataSet:
        currentLabel = entry[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    # shannonEnt = -shannonEnt
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 对连续变量划分数据集，direction规定划分的方向，
# 决定是划分出小于vlaue的数据样本还是大于value的数据样本集
def splitContinuousDataSet(dataSet, axis,value, direction):
    retDataSet=[]
    for featVec in dataSet:
        if direction==0:
            if featVec[axis]>value:
                reducedFeatVec=featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
        else:
            if featVec[axis]<=value:
                reducedFeatVec=featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
    return retDataSet




def chooseBestFeatureToSplit(dataSet, labels):
    numFeatures=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)
    bestInfoGain=0.0
    bestFeature = -1
    bestSplitDict={}
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        # 连续特征值判断
        if type(featList[0]).__name__=='float' or type(featList[0]).__name__=='int':
            # 产生n-1个候选划分点
            sortfeatList=sorted(featList)
            splitList=[]
            for j in range(len(sortfeatList)-1):
                splitList.append((sortfeatList[j]+sortfeatList[j+1])/2.0)
            bestSplitEntropy=10000
            slen=len(splitList)
            # 求用第j个候选划分点时，得到的信息熵，并记录最佳划分点
            for j in range(slen):
                # 获取每一个划分点
                value=splitList[j]
                newEntropy=0.0
                subDataSet0=splitContinuousDataSet(dataSet,i,value,0)
                subDataSet1=splitContinuousDataSet(dataSet,i,value,1)
                prob0=len(subDataSet0)/float(len(dataSet))
                newEntropy+=prob0*calcShannonEnt(subDataSet0)
                prob1=len(subDataSet1)/float(len(dataSet))
                newEntropy+=prob1*calcShannonEnt(subDataSet1)
                if newEntropy<bestSplitEntropy:
                    bestSplitEntropy=newEntropy
                    bestSplit=j
            # 用字典记录当前特征的最佳划分点
            bestSplitDict[labels[i]] = splitList[bestSplit]
            infoGain=baseEntropy-bestSplitEntropy
            
        # 对离散型特征值进行
        else:
            uniqueVals=set(featList)
            newEntropy=0.0
            # 计算该特征下每种划分的信息熵
            for value in uniqueVals:
                subDataSet = splitDataSet(dataSet, i, value)
                # print(f"subdataset: {subDataSet}")
                prob=len(subDataSet)/float(len(dataSet))
                newEntropy+=prob*calcShannonEnt(subDataSet)
            infoGain=baseEntropy-newEntropy
        if infoGain>bestInfoGain:
            bestInfoGain=infoGain
            bestFeature=i
    # 若当前节点的最佳划分特征为连续特征，则将其以之前记录的划分点为界进行二值化处理
    # 即是否小于等于bestSplitValue
    if type(dataSet[0][bestFeature]).__name__=='float' or type(dataSet[0][bestFeature]).__name__=='int':      
            bestSplitValue=bestSplitDict[labels[bestFeature]]        
            labels[bestFeature]=labels[bestFeature]+'<='+str(bestSplitValue)
            for i in range(len(dataSet)):
                if dataSet[i][bestFeature]<=bestSplitValue:
                    dataSet[i][bestFeature]=1
                else:
                    dataSet[i][bestFeature]=0


    return bestFeature



# 主程序，递归产生决策树
def createTree(dataSet, labels, data_full, labels_full):
    classList=[example[-1] for example in dataSet]
    # print(classList)
    # print(dataSet[0])
    # 统计正例的数量，如果等于所有的例子
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet, labels)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    if type(dataSet[0][bestFeat]).__name__=="str":
        currentlabel=labels_full.index(labels[bestFeat])
        featValuesFull=[example[currentlabel]for example in data_full]
        uniqueValuesFull=set(featValuesFull)
    del labels[bestFeat]
    # 针对bestFeat的每个取值，划分出一个子树
    for value in uniqueVals:
        subLabels=labels[:]
        if type(dataSet[0][bestFeat]).__name__=='str':
            uniqueValuesFull.remove(value)
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet, bestFeat, value), subLabels, data_full, labels_full)
    if type(dataSet[0][bestFeat]).__name__=='str':
        for value in uniqueValuesFull:
            myTree[bestFeatLabel][value]=majorityCnt(classList)
    return myTree


import matplotlib.pyplot as plt
decisionNode=dict(boxstyle="sawtooth", fc="0.8")
leafNode=dict(boxstyle="round4", fc="0.8")
arrow_args=dict(arrowstyle="<-")


# 计算树的叶子结点数量
def getNumLeafs(myTree):
    numLeafs=0
    # print(myTree.keys())
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs+=getNumLeafs(secondDict[key])
        else:
            numLeafs+=1
    return numLeafs


# 计算树的最大深度
def getTreeDepth(myTree):
    maxDepth=0
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else:thisDepth=1
        if thisDepth>maxDepth:
            maxDepth=thisDepth
    return maxDepth


# 画节点
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',\
    xytext=centerPt,textcoords='axes fraction',va="center", ha="center",\
    bbox=nodeType,arrowprops=arrow_args)
 
# 画箭头上的文字
def plotMidText(cntrPt, parentPt, txtString):
    lens=len(txtString)
    xMid=(parentPt[0]+cntrPt[0])/2.0-lens*0.002
    yMid=(parentPt[1]+cntrPt[1])/2.0
    createPlot.ax1.text(xMid, yMid, txtString)



def plotTree(myTree,parentPt,nodeTxt):
    numLeafs=getNumLeafs(myTree)
    depth=getTreeDepth(myTree)
    firstStr=list(myTree.keys())[0]
    cntrPt=(plotTree.x0ff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.y0ff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict=myTree[firstStr]
    plotTree.y0ff=plotTree.y0ff-1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.x0ff=plotTree.x0ff+1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.x0ff,plotTree.y0ff),cntrPt,leafNode)
            plotMidText((plotTree.x0ff,plotTree.y0ff),cntrPt,str(key))
    plotTree.y0ff=plotTree.y0ff+1.0/plotTree.totalD
 
def createPlot(inTree):
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW=float(getNumLeafs(inTree))
    plotTree.totalD=float(getTreeDepth(inTree))
    plotTree.x0ff=-0.5/plotTree.totalW
    plotTree.y0ff=1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.savefig("tree.png")  # ！！！保存结果要在show()之前，不然保存结果是白图
    plt.show()



if __name__ == "__main__":
    df = pd.read_csv("watermelon_4_3.csv",sep="\t")
    # print(df.values)
    data = df.values[:, 1:].tolist()
    data_full = data[:]
    # print(data_full)
    labels=df.columns.values[1:-1].tolist()
    labels_full=labels[:]
    # print(labels)
    # print(labels_full)
    # print(labels_full)
    myTree=createTree(data, labels, data_full, labels_full)
    print(myTree)
    # {'texture': {'little_blur': {'touch': {'hard_smooth': 0, 'soft_stick': 1}}, 
    # 'distinct': {'density<=0.3815': {0: 1, 1: 0}}, 'blur': 0}}


    # createPlot(myTree)
