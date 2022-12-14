"""
Function for Loss value
"""

import numpy as np

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)
from xDLUtils import Tools

"""
均方差损失函数
"""
class MseLoss:

    @staticmethod
    def loss(y,y_, n):
        corect_logprobs = Tools.mse(y, y_)
        data_loss = np.sum(corect_logprobs) / n
        delta = (y - y_) / n

        return data_loss, delta ,None

"""
二元交叉熵损失函数
"""
class SoftmaxCrossEntropyLoss:
    @staticmethod
    def loss(y,y_, n):
        y_argmax = np.argmax(y, axis=1)
        softmax_y = Tools.softmax(y)
        acc = np.mean(y_argmax == y_)
        # loss
        corect_logprobs = Tools.crossEntropy(softmax_y, y_)
        data_loss = np.sum(corect_logprobs) / n
        # delta
        softmax_y[range(n), y_] -= 1
        delta = softmax_y / n

        return data_loss, delta, acc, y_argmax
