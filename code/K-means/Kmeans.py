'''
Descroption:
Author: EmoryHuang
Date: 2021-07-02 16:27:43
Method:
'''

import numpy as np
import matplotlib.pyplot as plt


# 加载数据
def loadDataSet(file):
    dataSet = np.loadtxt(file, delimiter='\t')
    return dataSet


# 计算欧式距离
def euclDistance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))  # 计算欧氏距离


# 初始化 k 个聚类中心
def getCenter(dataSet, k):
    # 行，列大小
    m, n = dataSet.shape
    # 初始化 k 个聚类中心
    center = np.zeros((k, n))
    for i in range(k):
        # 产生 k 个 [0, m) 的数
        index = int(np.random.uniform(0, m))
        center[i, :] = dataSet[index, :]
    return center


# 样本点到最近的聚类中心的距离
def getClosestDist(data, center):
    min_dist = np.inf
    m = np.shape(center)[0]  # 当前已经初始化的聚类中心的个数
    for i in range(m):
        # 计算样本点与每个聚类中心之间的距离
        d = euclDistance(center[i, :], data)
        # 选择最短距离
        if min_dist > d:
            min_dist = d
    return min_dist


# 初始化 k 个聚类中心
def getCenterPlusPlus(dataSet, k):
    m, n = dataSet.shape
    # 初始化 k 个聚类中心
    center = np.zeros((k, n))
    # 1、随机选择一个样本点为第一个聚类中心
    index = np.random.randint(0, m)
    center[0, :] = dataSet[index, :]
    # 初始化一个距离的序列
    d = [0.0 for _ in range(m)]

    for i in range(1, k):
        sum_all = 0
        for j in range(m):
            # 2、对每一个样本找到最近的聚类中心点
            d[j] = getClosestDist(dataSet[j, ], center[0:i, ])
            # 将所有的最短距离相加
            sum_all += d[j]
        # 3、用轮盘法选出下一个聚类中心
        # 取得sum_all之间的随机值
        sum_all *= np.random.random()
        for j, dis in enumerate(d):
            sum_all -= dis
            if sum_all > 0:
                continue
            # 选择新的聚类中心
            center[i, :] = dataSet[j, :]
            break
    return center


# k均值聚类
def KMeans(dataSet, k):
    m = np.shape(dataSet)[0]  # 行的数目
    # 第一列存样本属于哪一类，初始为 0
    # 第二列存样本的到类的中心点的误差
    clusterAssment = np.mat(np.zeros((m, 2)))  # 创建 m 行 2 列的矩阵
    clusterChange = True  # 记录样本的点类是否发生变化

    # 第1步 初始化center
    center = getCenterPlusPlus(dataSet, k)
    # 如果上一次迭代过程中仍有样本点的类别发生变化，则继续计算
    while clusterChange:
        clusterChange = False

        # 遍历所有的样本（行数）
        for i in range(m):
            minDist = np.inf
            minIndex = -1

            # 第2步 找出离样本点最近的质心
            # 遍历所有质心
            for j in range(k):
                # 计算该样本到质心的欧式距离
                distance = euclDistance(center[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance  # 更新最短距离
                    minIndex = j  # 更新离该样本最近的中心

            # 第 3 步：更新每一行样本所属的类
            # 如果样本类别发生变化
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True
                # 更新类别以及误差
                clusterAssment[i, :] = minIndex, minDist**2

        # 第 4 步：更新质心
        # 遍历每一个类
        for j in range(k):
            # .A 将矩阵转化为数组
            # nonzero(a) 返回数组a中非零元素的索引值数组
            pointsInCluster = dataSet[np.nonzero(
                clusterAssment[:, 0].A == j)[0]]  # 获取类所有的点
            center[j, :] = np.mean(pointsInCluster, axis=0)   # 对矩阵的行求均值

    print("Congratulations,cluster complete!")
    return center, clusterAssment


def showCluster(dataSet, k, center, clusterAssment):
    m, n = dataSet.shape
    if n != 2:
        print("数据不是二维的")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("k值太大了")
        return 1

    # 绘制所有的样本
    for i in range(m):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 绘制质心
    for i in range(k):
        plt.plot(center[i, 0], center[i, 1], mark[i])
    plt.show()


dataSet = loadDataSet("D:/Document/code/Algorithm/K-means/test.txt")
k = 4
center, clusterAssment = KMeans(dataSet, k)

showCluster(dataSet, k, center, clusterAssment)
