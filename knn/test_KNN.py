#coding:utf-8

from KNN import *
from numpy import *


dataSet,labels=creatDataset()
#测试集合
testX=array([1.2,1.1])
K=3

outputLabelX=classify0(testX,dataSet,labels,K)

print 'input is:',testX,'output class is:',outputLabelX
