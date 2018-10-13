#!/usr/bin/python                                                                                                                                                                                                                    
#coding=utf-8

#File Name: test.py
#Author   : john
#Mail     : john.y.ke@mail.foxconn.com 
#Created Time: Sat 01 Sep 2018 05:38:56 PM CST

from numpy import *
import matplotlib.pyplot as plt
import pandas as pd

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))

def loadDataSet(fileName):
    dataMat = []        #特征数据集
    labelMat = []       #标签数据集
    fr = open(fileName) #打开原始数据集
    for line in fr.readlines():
        lineArr = line.strip().split('\t')  #分割特征
        dataMat.append([float(lineArr[0]), float(lineArr[1])])  #特征列表
        labelMat.append(float(lineArr[2]))  #标签列表
    return dataMat, labelMat #返回特征集与标签集

def selectJrand(i, m):  #选取随机数，i为alpha下标，m为alpha数目
    j = i
    while (j == i):
        j = int(random.uniform(0, m))    #产生一个0~m的随机数
    return j

def clipAlpha(aj, H, L):  #调整目标值
    if aj > H:  #目标值大于最大值
        aj = H
    if L > aj:  #目标值小于最小值
        aj = L
    return aj   #返回目标值

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):   #简单SMO算法
    dataMatrix = mat(dataMatIn) #特征值转为矩阵
    labelMat = mat(classLabels).transpose() #标签值转为矩阵，且转置
    m, n = shape(dataMatrix)    #特征值维度
    b = 0
    alphas = mat(zeros((m, 1))) #初始化m行1列的alpha向量全为0

    iter = 0
    while iter < maxIter:       #迭代次数
        alphaPairsChanged = 0   #记录alphas是否优化
        for i in range(m):
            #预测的类别 y[i] = w^Tx[i]+b; 其中因为 w = Σ(1~n) a[n]*label[n]*x[n]
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b   #预测结果
            Ei = fXi - float(labelMat[i])   #误差  预测结果 - 真实结果
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):   #误差较大，需要优化，正常值在(0~C)
                j = selectJrand(i, m)   #选择第二个随机alpha[j]
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b   #预测结果
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                #利用L和H将alpha调整到0~C
                if (labelMat[i] != labelMat[j]):    #异侧
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:#同侧
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L == H:  #已是最优，无需优化
                    #print("L == H")
                    continue

                #序列最小优化算法计算alpha[j]的最优值
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue

                alphas[j] -= labelMat[j] * (Ei - Ej) / eta  #计算出一个新的alphas[j]值
                alphas[j] = clipAlpha(alphas[j], H, L)      #对L和H进行调整
                if (abs(alphas[j] - alphaJold) < 0.00001):  #如果改变幅度较小，无需继续优化
                    #print("j not moving enough")
                    continue

                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                #print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        #print("iteration number: %d" % iter)

    return b, alphas#返回模型常量值和拉格朗日因子

def calcEk(oS, k):  #计算误差值
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b  #预测值
    Ek = fXk - float(oS.labelMat[k])    #误差 预测值-真实值

    return Ek

def selectJ(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    maxK = -1
    maxDeltaE = 0
    Ej = 0

    oS.eCache[i] = [1, Ei]  #首先将输入值Ei在缓存中设置成为有效的。这里的有效意味着它已经计算好了。

    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:  # 在所有的值上进行循环，并选择其中使得改变最大的那个值
            if k == i:
                continue

            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:  #如果是第一次循环，则随机选择一个alpha值
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
        return j, Ej

def updateEk(oS, k):  #计算误差值并存入缓存中
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):  #内循环代码
    Ei = calcEk(oS, i)

    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)  #选择最大的误差对应的j进行优化。效果更明显
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        #L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接return 0
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0

        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print("eta>=0")
            return 0

        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)    #使用辅助函数，以及L和H对其进行调整
        updateEk(oS, j) #更新误差缓存

        if (abs(oS.alphas[j] - alphaJold) < 0.00001):   #检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环。
            print("j not moving enough")
            return 0

        # 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)

        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter):    #完整SMO算法
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)  #创建一个 optStruct 对象
    iter = 0
    entireSet = True
    alphaPairsChanged = 0

    # 循环遍历：循环maxIter次 并且 （alphaPairsChanged存在可以改变 or 所有行遍历一遍）
    # 循环迭代结束 或者 循环遍历所有alpha后，alphaPairs还是没变化
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0

        #  当entireSet=true or 非边界alpha对没有了；就开始寻找 alpha对，然后决定是否要进行else。
        if entireSet:
            # 在数据集上遍历所有可能的alpha
            for i in range(oS.m):
                # 是否存在alpha对，存在就+1
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        # 对已存在 alpha对，选出非边界的alpha值，进行优化。
        else:
            # 遍历所有的非边界alpha值，也就是不在边界0或C上的值。
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1

        # 如果找到alpha对，就优化非边界alpha值，否则，就重新进行寻找，如果寻找一遍 遍历所有的行还是没找到，就退出循环。
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas

def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)

    return w

def plotfig_SVM(xMat, yMat, ws, b, alphas):
    xMat = mat(xMat)
    yMat = mat(yMat)

    # b原来是矩阵，先转为数组类型后其数组大小为（1,1），所以后面加[0]，变为(1,)
    b = array(b)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(xMat[:, 0].flatten().A[0], xMat[:, 1].flatten().A[0])

    x = arange(-1.0, 10.0, 0.1)     #x最大值，最小值根据原数据集dataArr[:, 0]的大小而定
    y = (-b-ws[0, 0]*x)/ws[1, 0]    #根据x.w + b = 0 得到，其式子展开为w0.x1 + w1.x2 + b = 0, x2就是y值
    ax.plot(x, y)

    for i in range(shape(yMat[0, :])[1]):
        if yMat[0, i] > 0:
            ax.scatter(xMat[i, 0], xMat[i, 1], color='blue')
        else:
            ax.scatter(xMat[i, 0], xMat[i, 1], color='green')

    # 找到支持向量，并在图中标红
    for i in range(100):
        if alphas[i] > 0.0:
            ax.scatter(xMat[i, 0], xMat[i, 1], color='red')
    plt.show()

if __name__ == '__main__':
    filePath = 'C:\\Users\\John\\Desktop\\DataBooks\\MachineLearning\\Ch06\\testSet.txt'
    dataMat, labelMat = loadDataSet(filePath)

    #b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)#简单SMO算法
    b, alphas = smoP(dataMat, labelMat, 0.6, 0.001, 40)    #完整SMO算法

    #print('b = ', b)
    #print('alphas[alphas>0] = ', alphas[alphas > 0])
    #print('shape(alphas[alphas > 0]) = ', shape(alphas[alphas > 0]))

    for i in range(100):
        if alphas[i] > 0:
            print(dataMat[i], labelMat[i])
    # 画图
    ws = calcWs(alphas, dataMat, labelMat)
    plotfig_SVM(dataMat, labelMat, ws, b, alphas)