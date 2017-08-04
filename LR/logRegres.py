#coding:utf-8

################
'''
author:zhangzhen
create_time:2017-6-23
by jiqixuexishizhan
'''
##########


from numpy import *



def loadDataSet(filename):
    dataMat=[];labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat


def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#梯度上升
def gradAscent(dataMatIn,classLabels):
    #测试集矩阵
    dataMatrix=mat(dataMatIn)
    #label矩阵
    labelMat=mat(classLabels).transpose()
    m,n=shape(dataMatrix)
    #学习率
    alpha=0.001
    #迭代次数
    maxCycles=500
    #权重初始值为1
    weights=ones((n,1))
    for k in range(maxCycles):
        #调用激活函数
        h=sigmoid(dataMatrix*weights)
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights

#随机梯度上升
def stocGradAscent0(dataMatrix,labelMat):
    dataMatrix=array(dataMatrix)
    m,n=shape(dataMatrix)
    alpha=0.01
    weights=ones(n)
    for  i in range(0,m):
        h=sigmoid(sum(dataMatrix[i]*weights))
        error=labelMat[i]-h
        weights=weights+alpha*error*dataMatrix[i]
    return weights

#改进随机梯度上升
def stocGradAscent1(dataMatrix,labelMat,numIter=150):
    m,n=shape(dataMatrix)
    weights=ones(n)
    for i in range(0,numIter):
        dataIndex=range(m)
        for j in range(0,m):
            #随机梯度
            alpha=4/(1.0+i+j)+0.01
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=labelMat[randIndex]-h
            weights=weights+alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])

    return weights



def plotBestFit(wei,dataMat,labelMat):

    import matplotlib.pyplot as plt
  #  dataMat,labelMat=loadDataSet(filename)
    weights=wei
    dataArr=array(dataMat)
    n=shape(dataArr)[0]
    xcords1=[];ycords1=[]
    xcords2=[];ycords2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcords1.append(dataArr[i,1]);ycords1.append(dataArr[i,2])
        else:
            xcords2.append(dataArr[i,1]);ycords2.append(dataArr[i,2])

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcords1,ycords1,s=30,c='red',marker='s')
    ax.scatter(xcords2,ycords2,s=30,c='blue')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]

    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()


def main():
    dataMatrix,labelMat=loadDataSet('test_0.txt')
    #weight=stocGradAscent0(array(dataMatrix),labelMat)
    #print weight
    weight=stocGradAscent1(array(dataMatrix),labelMat)
    print weight
    plotBestFit(weight,dataMatrix,labelMat)



if __name__ == '__main__':
    main()



