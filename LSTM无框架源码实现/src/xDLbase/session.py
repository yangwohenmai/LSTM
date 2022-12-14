"""
fit model
"""

from unittest import result
import numpy as np

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)
from xDLUtils import Tools
from lossfunc import *

# 会话类

class Session(object):

    def __init__(self, layers, lossfunCls):
        # 设置模型的网络层
        self.layers = layers
        self.input = []
        # 设置损失函数
        self.lossCls = lossfunCls

    # 前向传播算法
    # val = False 训练 ， True 测试/预测
    def inference(self, train_data, y_, val=False):
        # 当前batch中的sample数量, y(32,)
        curr_batch_size = len(y_)
        # train_data(32, 1, 2)
        self.input = train_data
        # 首次dataLayer记录的是输入矩阵train_data，后续记录每一层网络层运算后的输出矩阵，作为下一个网络层的输入矩阵
        dataLayer = train_data

        # 前向传播
        if False == val:
            # self.layers包含两部分，第一部分是3个lstm层，第二部分是全连接层
            for layer in self.layers:
                # 此处返回的dataLayer是上一个网络层输出的状态h，作为下一个网络层的输入
                dataLayer = layer.fp(dataLayer)

        else:  # 预测/测试
            for layer in self.layers:
                dataLayer = layer.inference(dataLayer)
        
        # y记录全连接层的最终输出矩阵，y即预测值
        y = dataLayer
        # 先通过损失函数计算预测值y和真实值y_之间的loss值(均方差)；再计算误差矩阵(y_-y)，将误差均分在每个sample上，误差矩阵用于后续反向传播
        data_loss,delta,acc,result = self.lossCls.loss(y,y_,curr_batch_size)

        return y, data_loss, delta, acc, result
    
    # 反向传播算法
    def bp(self, delta, lrt):
        # 传入delta为损失误差矩阵，deltaLayer为后续反向传播时每一层迭代计算后的误差矩阵
        deltaLayer = delta

        """此处多此一举，先用for循环对fcnn层求反向传播，而后直接写layers[0]对rnn层求反向传播"""
        # 反向传播，用reversed对网络层顺序反转
        for i in reversed(range(1, len(self.layers))):
            # 传入误差矩阵deltaLayer，以及上一层的权重矩阵self.layers[i - 1].out（后续没用到）
            deltaLayer = self.layers[i].bp(self.layers[i - 1].out, deltaLayer, lrt)
        # deltaLayer是全连接层反向出入的误差矩阵
        self.layers[0].bp(self.input, deltaLayer, lrt)

    # 实现训练步骤
    def train_steps(self, train_data, y_, lrt):
        # 前向传播，返回delta误差矩阵
        _,loss, delta,acc,result = self.inference(train_data, y_, val=False)
        # 反向传播，传入delta误差矩阵
        self.bp(delta, lrt)
        return acc, loss

    # 独立数据集验证训练结果
    def validation(self, data_v, y_v):
        y, loss,_,acc,result = self.inference(data_v, y_v, val=True)
        return y, loss , acc, result