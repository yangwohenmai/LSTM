"""
Fully Connected Layer Class
"""

import numpy as np

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)
from xDLUtils import Tools

#Tools = xnnUtils.Tools()
# 全连接类
class FCLayer(object):
    def __init__(self,miniBatchesSize,i_size,o_size,activator,optimizerCls,optmParams,needReshape,dataType,init_w):
        # 初始化超参数
        self.miniBatchesSize = miniBatchesSize
        # 输入尺寸
        self.i_size = i_size
        # 输出尺寸
        self.o_size = o_size
        self.activator = activator
        # 初始化全连接层优化器
        self.optimizerObj = optimizerCls(optmParams, dataType)
        # 是否将N,T,D输入，先拉伸成N,T*D,再做仿射变换
        # 在letNet-5中，可以将pooling层输出3维拉成2维
        # 在RNN中，可以将N v M中的N个T时刻输出D维向量，变成N,T*D ,再仿射变换 为 N*D' 规格, 起到 N->M映射的效果
        self.needReshape = needReshape
        self.dataType = dataType
        # 初始化全连接层的权重矩阵 w->shape(300,10)，i_size=timesteps*hidden
        self.w = init_w * np.random.randn(i_size, o_size).astype(dataType)
        # 初始化偏置矩阵 w->shape(10,)
        self.b = np.zeros(o_size, dataType)
        self.out = []
        # 通过本层权重反向传播
        self.deltaPrev = [] #上一层激活后的误差输出
        self.deltaOri = [] #本层原始误差输出
        self.shapeOfOriIn = () #原始输入维度
        self.inputReshaped =[]
    # 预测时前向传播
    def inference(self, input):
        self.shapeOfOriIn = input.shape
        self.out = self.fp(input)
        return self.out

    # 全连接层的前向传播,激活后再输出
    def fp(self, input):
        # 全连接层首先对输入进行拉伸变形处理，相当于Flatten()的功能，将(32,10,30)->(32,300)
        self.shapeOfOriIn = input.shape
        self.inputReshaped = input if self.needReshape is False else input.reshape(input.shape[0],-1)
        # 先将输入矩阵与全连接层权重矩阵相乘，再进行激活函数运算
        self.out = self.activator.activate(Tools.matmul(self.inputReshaped, self.w) + self.b)
        ####debug####
        # np.savetxt('G:/0tmp/0debug/x.csv',self.inputReshaped[0])
        # np.savetxt('G:/0tmp/0debug/w_c1.csv', self.w[:,0])
        # np.savetxt('G:/0tmp/0debug/w_c2.csv', self.w[:, 1])
        # np.savetxt('G:/0tmp/0debug/out.csv', self.out[0])
        ####debug end#####
        return self.out

    # 反向传播方法(误差和权参)
    def bp(self, input, delta, lrt):
        # 先将输入的误差矩阵通过激活函数求导计算
        self.deltaOri = self.activator.bp(delta, self.out)
        # # 恢复拉伸变形
        # self.deltaOri = deltaOri_reshaped if self.needReshape is False else deltaOri_reshaped.reshape(self.shapeOfOriIn)
        # 将误差矩阵与上一层权重矩阵的转置做乘法，可将误差反向传播至上一层
        self.bpDelta()
        # 通过优化器对反向传播进行梯度优化
        self.bpWeights(input, lrt)
        return self.deltaPrev

    # 误差矩阵反向传播
    def bpDelta(self):
        # 将通过激活函数求导后的误差矩阵deltaOri，和当前FCNN层的权重矩阵转置w.T相乘，来将误差向前传播
        deltaPrevReshapped = Tools.matmul(self.deltaOri, self.w.T)
        # 误差矩阵恢复之前Flatten()的拉伸变形，适配上层网络shape以便向上层网络继续反向传播
        self.deltaPrev = deltaPrevReshapped if self.needReshape is False else deltaPrevReshapped.reshape(self.shapeOfOriIn)
        return self.deltaPrev

    # 计算反向传播权重梯度w,b
    def bpWeights(self, input, lrt):
        # dw = Tools.matmul(input.T, self.deltaOri)
        # inputReshaped是正向传播时上层网络传到本层的输入矩阵，deltaOri是本层反向传播激活函数求导后的误差矩阵（此处的数学原理？）
        dw = Tools.matmul(self.inputReshaped.T, self.deltaOri)
        # 误差矩阵deltaOri.shape->(32,10),进行sum计算后将32个sample向量对应位置求和后db.shape->(1,10)，再reshape成b.shape用于后续梯度优化
        db = np.sum(self.deltaOri, axis=0, keepdims=True).reshape(self.b.shape)
        # 当前层网络权重(w,b)元组
        weight = (self.w,self.b)
        # 反向传播的权重(dw,db)元组
        dweight = (dw,db)
        # 元组按引用对象传递，值在方法内部已被更新
        # 将两个元组代入优化器，求出梯度方向，根据梯度更新参数，更新了self.lstmParams[i][Wx,Wh,b]的权重矩阵
        self.optimizerObj.getUpdWeights(weight,dweight, lrt)