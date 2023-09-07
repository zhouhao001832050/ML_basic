# -*- coding: utf-8 -*-
import pandas as pd
import copy

def calcGini(dataSet):
    numEntries=len(dataSet)
    labelCounts={}
    # 给所有可能分类创建字典
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    Gini=1.0
    # 以2为底数计算香农熵
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        Gini-=prob*prob
    return Gini


def splitDataSet(dataSet, axis, value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def splitContinuousDataSet(dataSet, axis, value, direction):
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
    bestGiniIndex=100000.0 # 由于要找出的是基尼系数最小的，所以要定义一个大的数
    bestFeature=-1
    bestSplitDict={}
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]
        # 对连续型特征进行处理
        if type(featList[0]).__name__=="float"or type(featList[0]).__name__=="int":
            # 产生n-1个候选划分点
            sortfeatList=sorted(featList)
            splitList=[]
            for j in range(len(sortfeatList)-1):
                splitList.append((sortfeatList[j]+sortfeatList[j+1])/2.0)
            bestSplitGini=10000
            slen=len(splitList)
            # 求用第j个候选划分点划分时，得到的信息熵，并记录最佳划分点
            for j in range(slen):
                value=splitList[j]
                newGiniIndex=0.0
                subDataSet0=splitContinuousDataSet(dataSet, i, value, 0)
                subDataSet1=splitContinuousDataSet(dataSet, i, value, 1)
                prob0=len(subDataSet0)/float(len(dataSet))
                newGiniIndex+= prob0*calcGini(subDataSet0)
                prob1=len(subDataSet1)/float(len(dataSet))
                newGiniIndex+= prob1*calcGini(subDataSet1)
                if newGiniIndex<bestSplitGini:
                    bestSplitGini=newGiniIndex
                    bestSplit=j
            # 用字典记录当前特征的最佳划分点
            bestSplitDict[labels[i]]=splitList[bestSplit]
            GiniIndex=bestSplitGini
        # 对离散型特征进行处理
        else:
            uniqueVals=set(featList)
            newGiniIndex=0.0
            # 计算该特征下每种划分的信息熵
            for value in uniqueVals:
                subDataSet=splitDataSet(dataSet,i,value)
                prob=len(subDataSet)/float(len(dataSet))
                newGiniIndex+=prob*calcGini(subDataSet)
            GiniIndex=newGiniIndex
        if GiniIndex<bestGiniIndex:
            bestGiniIndex=GiniIndex
            bestFeature=i
    # 若当前节点的最佳划分特征为连续特征，则将其之前记录的划分点为界进行二值化处理
    # 即是否小于等于bestSplitValue
    if type(dataSet[0][bestFeature]).__name__=='float' or type(dataSet[0][bestFeature]).__name__=='int':      
        bestSplitValue=bestSplitDict[labels[bestFeature]]
        labels[bestFeature]=labels[bestFeature]+"<="+str(bestSplitValue)
        for i in range(len(dataSet)):
            if dataSet[i][bestFeature]<=bestSplitValue:
                dataSet[i][bestFeature]=1
            else:
                dataSet[i][bestFeature]=0
    return bestFeature


# 若特征已经划分完，节点下的样本还没有统一取值，则需要进行投票
def majorityCnt(calssList):
    classCount={}
    for vote in calssList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    return max(classCount)


def testing(myTree,data_test,labels):
    error=0.0
    for i in range(len(data_test)):
        if classify(myTree, labels, data_test[i])!=data_test[i][-1]:
            error+=1
    # print(f"myTree:{error}")
    return float(error)


def testingMajor(major, data_test):
    error=0.0
    for i in range(len(data_test)):
        if major!=data_test[i][-1]:
            error+=1
    # print(f"major:{error}")
    return float(error)


# 由于在Tree中，连续特征的名称以及改为了 feature<=value的形式
# 因此对于这类特征，需要利用正则表达式进行分割，获得特证明以及分割阈值
def classify(inputTree, featLabels, testVec):
    firstStr=list(inputTree.keys())[0]
    # print(f"firstSre:{firstStr}")
    # print(inputTree)
    if "<=" in firstStr:
        featvalue=float(re.compile("(<=.+)").search(firstStr).group()[2:])
        featkey=re.compile("(.+<=)").search(firstStr).group()[:-2]
        secondDict=inputTree[firstStr]
        featIndex=featLables.index(featkey)
        if testVec[featIndex]<=featvalue:
            judge=1
        else:
            judge=0
        for key in secondDict.keys():
            if judge==int(key):
                if type(secondDict[key]).__name__=="dict":
                    classLabel=classify(secondDict[key], featLabels, testVec)
                else:
                    classLabel=secondDict[key]
    else:
        secondDict=inputTree[firstStr]
        featIndex=featLabels.index(firstStr)
        for key in secondDict.keys():
            if testVec[featIndex]==key:
                if type(secondDict[key]).__name__=="dict":
                    classLabel=classify(secondDict[key], featLabels, testVec)
                else:
                    classLabel=secondDict[key]
    return classLabel


#后剪枝
def postPruningTree(inputTree,dataSet,data_test,labels):
    firstStr=inputTree.keys()[0]
    secondDict=inputTree[firstStr]
    classList=[example[-1] for example in dataSet]
    featkey=copy.deepcopy(firstStr)
    if '<=' in firstStr:
        featkey=re.compile("(.+<=)").search(firstStr).group()[:-2]
        featvalue=float(re.compile("(<=.+)").search(firstStr).group()[2:])
    labelIndex=labels.index(featkey)
    temp_labels=copy.deepcopy(labels)
    del(labels[labelIndex])
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            if type(dataSet[0][labelIndex]).__name__=='str':
                inputTree[firstStr][key]=postPruningTree(secondDict[key],\
                 splitDataSet(dataSet,labelIndex,key),splitDataSet(data_test,labelIndex,key),copy.deepcopy(labels))
            else:
                inputTree[firstStr][key]=postPruningTree(secondDict[key],\
                splitContinuousDataSet(dataSet,labelIndex,featvalue,key),\
                splitContinuousDataSet(data_test,labelIndex,featvalue,key),\
                copy.deepcopy(labels))
    if testing(inputTree,data_test,temp_labels)<=testingMajor(majorityCnt(classList),data_test):
        return inputTree
    return majorityCnt(classList)
 

# 主程序，递归产生决策时
def createTree(dataSet, labels, data_full, labels_full,data_test):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    temp_labels=copy.deepcopy(labels)
    bestFeat=chooseBestFeatureToSplit(dataSet, labels)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    # 取出最佳节点的值
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    if type(dataSet[0][bestFeat]).__name__=="str":
        currentlabel=labels_full.index(labels[bestFeat])
        featValuesFull=[example[currentlabel] for example in data_full]
        uniqueValsFull=set(featValuesFull)
    del(labels[bestFeat]) # 去除掉该最佳节点
    # 针对bestFeat的每个取值，划分出一个子树
    for value in uniqueVals:
        subLabels=labels[:]
        if type(dataSet[0][bestFeat]).__name__=="str":
            uniqueValsFull.remove(value)
            myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet, bestFeat, value), 
                                                    subLabels, data_full, labels_full,splitDataSet(dataSet, bestFeat, value))
    if type(dataSet[0][bestFeat]).__name__=="str":
        for value in uniqueValsFull:
            myTree[bestFeatLabel][value]=majorityCnt(classList)
    # import pdb;pdb.set_trace()
    if testing(myTree, data_test, temp_labels)<testingMajor(majorityCnt(classList),data_test):
        return myTree
    return majorityCnt(classList)

import matplotlib.pyplot as plt
decisionNode=dict(boxstyle="sawtooth", fc="0.8")
leafNode=dict(boxstyle="round4", fc="0.8")
arrow_args=dict(arrowstyle="<-")


# 计算树的叶子结点数量
def getNumLeafs(myTree):
    numLeafs=0
    # print(myTree)    
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
    plt.savefig("cart_tree.png")  # ！！！保存结果要在show()之前，不然保存结果是白图
    plt.show()



if __name__ == "__main__":
    df = pd.read_csv("watermelon_4.4.csv", sep="\t")
    data = df.values[:11,1:].tolist()  # [['dark_green', 'curl_up', 'little_heavily', 'distinct', 'sinking', 'hard_smooth', 1], 
                                        # ['dark_green', 'curl_up', 'heavily', 'distinct', 'sinking', 'hard_smooth', 1]]
    data_full=data[:]
    data_test=df.values[11:,1:].tolist()
    labels=df.columns.values[1:-1].tolist()  # ['color', 'root', 'knocks', 'texture', 'navel', 'touch']
    labels_full=labels[:]
    # import pdb;pdb.set_trace()
    myTree=createTree(data, labels, data_full, labels_full, data_test)
    #  {'texture': {'little_blur': {'touch': {'hard_smooth': 0, 'soft_stick': 1}}, 
    # 'distinct': {'density<=0.3815': {0: 1, 1: 0}}, 'blur': 0}}
    print(myTree)
    createPlot(myTree)
