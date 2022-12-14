"""
Activator Function
"""

import numpy as np

# ReLU  Activator
class ReLU(object):
    @staticmethod
    def activate(x):
        return np.maximum(0, x)

    @staticmethod
    def bp(delta, x):
        delta[x <= 0] = 0
        return delta


# sigmoid  Activator
class Sigmoid(object):
    def __init__(self):
        self.o = []

    def activate(self, x):
        self.o = 1. / (1 + np.exp(-x))
        return self.o

    # 原函数直接得到导数， 形参和其它激活函数保持统一。
    def bp(self, _1, _2):
        return self.o(1 - self.o)


# xtanh  Activator
class Tanh(object):
    def __init__(self):
        self.o = []

    def activate(self, x):
        self.o = np.tanh(x)
        return self.o

    #def bp(self, _1, _2):
    def bp(self):
        return 1 - self.o * self.o


# 原样输出，不添加激活函数
class NoAct(object):
    @staticmethod
    def activate(x):
        return x

    @staticmethod
    def bp(delta, x):
        return delta
