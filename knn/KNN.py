#coding:utf-8
'''
机器学习实战：KNN算法
author：zhangzhen
2017-6-20

'''
from numpy import *
import operator

def  creatDataset():
    #测试数据集
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,K):
    #构造分类器
    dataSetSize=dataSet.shape[0]
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**0.5
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort()
    classCount={}
    for i  in range(K):
        voteLabel=labels[sortedDistIndicies[i]]
        classCount[voteLabel]=classCount.get(voteLabel,0)+1
    sortedClasscount=sorted(classCount.iteritems(),
        key=operator.itemgetter(1),reverse=True)
    return sortedClasscount[0][0]

