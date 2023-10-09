import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

####----------加载波士顿房价预测数据集-------------------######
data_url = "http://lib.stat.cmu.edu/datasets/boston"   ####数据所在的url
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None) ###将数据读入
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])  ###将X的值划分给data
target = raw_df.values[1::2, 2]           ###将Y的值划分给target

#-------------数据处理-------------------------------######

_x,_y=[],[]
_y=target
for item in data.tolist():
    item.append(float(1))
    _x.append(item)                 ##将训练样本的维度扩展一个为1的维度

#-------------样本数据参差不齐，我们对其进行归一化处理--------------###

class Feature_scaling:

    def __init__(self,_dataset,_target):
        self._dataset=_dataset
        self._target=_target

    def MaxMin(self):
        #最大值最小值缩放
        #这里有一些数据的值相差过大，不太容易把他们变成0，1范围内的数值
        #所以这里把他们缩放到10-20范围内
        _x=np.array(self._dataset)
        _y=np.array(self._target)
        _x=_x.T
        _x_Fixed=[]
        for item in _x:
            #让数据的每一列除以他们的最大值最小值之差
            if (np.max(np.array(item))-np.min(np.array(item)))==0:
                _x_Fixed.append(item)
            else:
                _x_Fixed.append(((item-np.min(np.array(item)))/(np.max(np.array(item))-np.min(np.array(item))))*10+10)
            _x=np.array(_x_Fixed).T
        return _x,_y

    def Mean_normalization(self):
        '''# 均值缩放
           # 这里有一些数据的值相差过大，不太容易把他们变成0，1范围内的数值
           # 同样的这里把他们缩放到10-20范围内
           # 这个和上面的最大最小值缩放没有多大区别'''
        _x = np.array(self._dataset)
        _y = np.array(self._target)
        _x = _x.T
        _x_Fixed = []
        for item in _x:
            if (np.max(np.array(item)) - np.min(np.array(item))) == 0:
                _x_Fixed.append(item)   #最后一列是我们加上去用来拟合B的，这里不做考虑对其缩放
            else:
                _x_Fixed.append(((item - np.mean(np.array(item))) / (np.max(np.array(item)) - np.min(np.array(item)))) * 10 + 10)
            _x = np.array(_x_Fixed).T
        return _x, _y

    def Z_score_Normalization(self):
        '''这种标准化方法也叫高斯归一化，
        他就是将数据的每一个特征值
        转换为均值为0
        标准差为1的数据'''
        _x_Fixed=[]
        _x=np.array(self._dataset)
        _y=np.array(self._target)
        for item in _x.T:
            avg=np.mean(np.array(item))  #求均值
            avr=np.var(np.array(item))   #求方差
            if (np.max(np.array(item)) - np.min(np.array(item))) == 0:
                _x_Fixed.append(item)   #最后一列是我们加上去用来拟合B的，这里不做考虑对其缩放
            else:
                _x_Fixed.append((np.array(item)-avg)/avr)
        _x=np.array(_x_Fixed).T

        return _x,_y
    def Scaling_to_unit_length(self):
        '''该方法，就是用当前特征值，除以
        当前特征维度的欧几里得长度，
        其实就是长度，不同情况不同分析把'''
        _x_Fixed=[]
        _x=np.array(self._dataset)
        _y=np.array(self._target)
        for item in _x.T:

            if (np.max(np.array(item)) - np.min(np.array(item))) == 0:
                _x_Fixed.append(item)   #最后一列是我们加上去用来拟合B的，这里不做考虑对其缩放
            else:
                #为了计算方便，我们取它的无穷范数
                _x_Fixed.append((np.array(item))/np.max(np.array(item)))
                #_x_Fixed.append((np.array(item)) / 10)
        _x=np.array(_x_Fixed).T

        return _x,_y

#-------------
def fit(_x,_y,mode="Grad"):
    if mode=="Grad":
        #####----定义一些需要的数据-----#####
        random.seed(100)
        learningRate = 0.00000001  # 初始化学习率
        weight = []
        for i in range(len(data[0])):
            weight.append(random.randint(1, 10))  # 为权重赋值
        weight.append(random.randint(1, 10))  ##将weight的维度扩展1，方便进行bais的运算
        #####----开始梯度下降---------#######
        weight=np.matrix(weight)   ########我们创建了一个行向量，
        _x=np.matrix(_x)
        _y=np.matrix(_y)
        cost,precost=1,0
        while cost>0.01:
            weight=weight-learningRate*(weight*_x.T-_y)*_x  ###梯度下降
            cost=np.sqrt((weight*_x.T-_y)*(weight*_x.T-_y).T)
            cost=cost/len(_y)
            print(cost)
            if abs(cost-precost)<0.00001:
                break
            precost=cost
        return weight

    if mode=='Newton':
        random.seed(100)
        learningRate = 1  # 初始化步长
        weight = []
        for i in range(len(data[0])):
            weight.append(random.randint(1, 10))  # 为权重赋值
        weight.append(random.randint(1, 10))  ##将weight的维度扩展1，方便进行bais的运算
        '''海瑟矩阵不知道怎么求了，
            本来以为我能求出来的，
            其实时间充裕一点还是
            能求出来的，但是时间有
            点紧张，所以就不做这种
            优化方法了
    '''

    if mode=="Normalize":
        x=np.array(_x)
        y=np.array(_y)
        #计算公式：W=（XTX）(-1)XTY
        weight=np.linalg.inv((x.T@x))@x.T@y
        return weight

mode='Normalize'
#################未缩放###########################
weight=fit(_x,_y,mode=mode)
weight=np.matrix(weight)   ########我们创建了一个行向量，
####---------------------画图-----------------######
_x=np.matrix(_x)
_YPredict=weight*_x.T          ##预测出的y的值
maxdata=np.max(_y)             #取y的最大值和最小值
mindata=np.min(_y)
length=maxdata-mindata          #获取最大值和最小值的长度
_y=np.sort(_y)
a=np.arange(0,length,length/len(_y))
_yPredict=np.sort(_YPredict.tolist()[0])

ClassFeature=Feature_scaling(_x,_y)
#########最大最小值缩放#################################
_x,_y=ClassFeature.MaxMin()
weight=fit(_x,_y,mode=mode)
weight=np.matrix(weight)   ########我们创建了一个行向量，
####---------------------画图-----------------######
_x=np.matrix(_x)
_YPredict=weight*_x.T          ##预测出的y的值
maxdata=np.max(_y)             #取y的最大值和最小值
mindata=np.min(_y)
length=maxdata-mindata          #获取最大值和最小值的长度
_y=np.sort(_y)
a=np.arange(0,length,length/len(_y))
_yPredict_MAXMIN=np.sort(_YPredict.tolist()[0])
###################################################
#################均值缩放###########################
_x,_y=ClassFeature.Mean_normalization()
weight=fit(_x,_y,mode=mode)
weight=np.matrix(weight)   ########我们创建了一个行向量，
####---------------------画图-----------------######
_x=np.matrix(_x)
_YPredict=weight*_x.T          ##预测出的y的值
maxdata=np.max(_y)             #取y的最大值和最小值
mindata=np.min(_y)
length=maxdata-mindata          #获取最大值和最小值的长度
_y=np.sort(_y)
a=np.arange(0,length,length/len(_y))
_yPredict_Mean=np.sort(_YPredict.tolist()[0])

#################均值缩放###########################
_x,_y=ClassFeature.Z_score_Normalization()
weight=fit(_x,_y,mode=mode)
weight=np.matrix(weight)   ########我们创建了一个行向量，
####---------------------画图-----------------######
_x=np.matrix(_x)
_YPredict=weight*_x.T          ##预测出的y的值
maxdata=np.max(_y)             #取y的最大值和最小值
mindata=np.min(_y)
length=maxdata-mindata          #获取最大值和最小值的长度
_y=np.sort(_y)
a=np.arange(0,length,length/len(_y))
_yPredict_Gass=np.sort(_YPredict.tolist()[0])

#################均值缩放###########################
_x,_y=ClassFeature.Scaling_to_unit_length()
weight=fit(_x,_y,mode=mode)
weight=np.matrix(weight)   ########我们创建了一个行向量，
####---------------------画图-----------------######
_x=np.matrix(_x)
_YPredict=weight*_x.T          ##预测出的y的值
maxdata=np.max(_y)             #取y的最大值和最小值
mindata=np.min(_y)
length=maxdata-mindata          #获取最大值和最小值的长度
_y=np.sort(_y)
a=np.arange(0,length,length/len(_y))
_yPredict_Length=np.sort(_YPredict.tolist()[0])

#_yPredict=_YPredict.tolist()[0]
plt.figure(figsize=(10, 10), dpi=50)
#plt.scatter(a, _y,color='red',label='Real')
#plt.scatter(a,_yPredict,color='green',label='Prediect')
plt.plot(a, _y,color='red',label='Source')
plt.plot(a,_yPredict,color='purple',label='NoNormalize')
plt.plot(a,_yPredict_MAXMIN,color='green',label='MAXMIN')
plt.plot(a,_yPredict_Mean,color='blue',label='MEAN')
plt.plot(a,_yPredict_Gass,color='pink',label='Gauss')
plt.plot(a,_yPredict_Length,color='yellow',label='SCORE')
plt.legend(['Source','NoNormalize','MAXMIN','MEAN','Gauss','SCORE'])
plt.show()