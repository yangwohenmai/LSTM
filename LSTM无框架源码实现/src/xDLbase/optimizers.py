"""
Some different optimizers class
"""

import numpy as np

class SGDOptimizer(object):
    def __init__(self, optmParams, dataType):
        # self.gamma, self.eps = optmParams
        self.dataType = dataType
        # self.isInited = False
        # self.v=[]
        # self.Iter = 0

    # # lazy init
    # def initV(self, w):
    #     if (False == self.isInited):
    #         for i in range(len(w)):
    #             self.v.append(np.zeros(w[i].shape, dtype=self.dataType))
    #         self.isInited = True

    def getUpdWeights(self, w, dw, lr):
        # self.initV(w)

        # t = self.Iter + 1
        wNew = []
        for i in range(len(w)):
            wi = self.OptimzSGD(w[i], dw[i], lr)
            wNew.append(wi)

        # 转为元组输出
        return tuple(wNew)

    def OptimzSGD(self, x, dx,lr):
        # v = self.gamma * v + lr * dx
        x += - lr * dx

        return x

# Momentum优化类
class MomentumOptimizer(object):
    def __init__(self, optmParams, dataType):
        self.gamma, self.eps = optmParams
        self.dataType = dataType
        self.isInited = False
        self.v=[]
        # self.Iter = 0

    # lazy init
    def initV(self, w):
        if (False == self.isInited):
            for i in range(len(w)):
                self.v.append(np.zeros(w[i].shape, dtype=self.dataType))
            self.isInited = True

    def getUpdWeights(self, w, dw, lr):
        self.initV(w)

        # t = self.Iter + 1
        wNew = []
        for i in range(len(w)):
            wi, self.v[i] = self.OptimzMomentum(w[i], dw[i], self.v[i], lr)
            wNew.append(wi)

        # 转为元组输出
        return tuple(wNew)

    def OptimzMomentum(self, x, dx, v, lr):
        v = self.gamma * v + lr * dx
        x += - v

        return x,  v

# Nesterov's Accelerated Gradient
class NAGOptimizer(object):
    def __init__(self, optmParams, dataType):
        self.gamma, self.eps = optmParams
        self.dataType = dataType
        self.isInited = False
        self.v=[]

    # lazy init
    def initV(self, w):
        if (False == self.isInited):
            for i in range(len(w)):
                self.v.append( np.zeros(w[i].shape, dtype=self.dataType))
            self.isInited = True

    # w和dw都是元组类型
    def getUpdWeights(self, w, dw, lr):
        self.initV(w)
        wNew = []
        for i in range(len(w)):
            wi, self.v[i] = self.OptimzNAG(w[i], dw[i], self.v[i], lr)
            wNew.append(wi)

        # 转为元组输出
        return tuple(wNew)

    def OptimzNAG(self, x, dx, v,lr):

        vt = self.gamma * v + lr * dx
        x += self.gamma* v - (1.+self.gamma) * vt

        return x, vt


# 自适应学习率优化，完成更新
class AdagradOptimizer(object):
    def __init__(self, optmParams, dataType):

        self.eps = optmParams
        self.dataType = dataType
        self.isInited = False
        self.g=[]

    # lazy init
    def initG(self, w):
        if (False == self.isInited):
            for i in range(len(w)):
                self.g.append( np.zeros(w[i].shape, dtype=self.dataType))
            self.isInited = True

    # w和dw都是元组类型
    def getUpdWeights(self, w, dw, lr):
        self.initG(w)
        wNew = []
        for i in range(len(w)):
            wi, self.g[i] = self.OptimzAdagrad(w[i], dw[i], self.g[i], lr)
            wNew.append(wi)

        # 转为元组输出
        return tuple(wNew)

    def OptimzAdagrad(self, x, dx, g,lr):

        g += dx ** 2
        x += - lr * dx / (np.sqrt(g) + self.eps)

        return x, g

# RMSprop优化，完成更新
class RMSpropOptimizer(object):
    def __init__(self, optmParams, dataType):
        self.gamma, self.eps = optmParams
        self.dataType = dataType
        self.isInited = False
        # self.g=[]
        self.eg = []

    # lazy init
    def initEG(self, w):
        if (False == self.isInited):
            for i in range(len(w)):
                # self.g.append( np.zeros(w[i].shape, dtype=self.dataType))
                self.eg.append(np.zeros(w[i].shape, dtype=self.dataType))
            self.isInited = True

    # w和dw都是元组类型
    def getUpdWeights(self, w, dw, lr):
        self.initEG(w)
        wNew = []
        for i in range(len(w)):
            wi, self.eg[i] = self.OptimzRMSprop(w[i], dw[i], self.eg[i], lr)
            wNew.append(wi)

        # 转为元组输出
        return tuple(wNew)

    def OptimzRMSprop(self, x, dx, eg,lr):

        eg =self.gamma * eg + (1-self.gamma) * (dx** 2)

        x += - lr * dx / (np.sqrt(eg) + self.eps)

        return x, eg

# RMSprop优化，完成更新
class AdaDeltaOptimizer(object):
    def __init__(self, optmParams, dataType):
        self.gamma, self.eps = optmParams
        self.dataType = dataType
        self.isInited = False
        self.eg=[]
        self.etsq = []
        self.dt = []

    # lazy init
    def initE(self, w):
        if (False == self.isInited):
            for i in range(len(w)):
                self.eg.append(np.zeros(w[i].shape, dtype=self.dataType))
                self.etsq.append(np.zeros(w[i].shape, dtype=self.dataType))
                self.dt.append(np.zeros(w[i].shape, dtype=self.dataType))
            self.isInited = True

    # w和dw都是元组类型
    def getUpdWeights(self, w, dw, lr):
        self.initE(w)
        wNew = []
        for i in range(len(w)):
            wi, self.eg[i],self.etsq[i],self.dt[i] = self.OptimzAdaDelta(w[i], dw[i], self.eg[i],self.etsq[i],self.dt[i])
            wNew.append(wi)

        # 转为元组输出
        return tuple(wNew)

    def OptimzAdaDelta(self, x, dx, eg,etsq, dt):

        eg =self.gamma * eg + (1-self.gamma) * (dx** 2)

        etsq=self.gamma * etsq + (1-self.gamma) * (dt** 2)
        dt = np.sqrt( (etsq + self.eps)/(eg +self.eps)) * dx
        x += - dt

        return x, eg,etsq,dt


# 自适应矩估计优化类
class AdamOptimizer(object):
    def __init__(self, optmParams, dataType):
        self.beta1 ,self.beta2 , self.eps = optmParams
        self.dataType = dataType
        self.isInited = False
        self.m=[]
        self.v=[]
        # self.m_w = []
        # self.v_w = []
        # self.m_b = []
        # self.v_b = []
        self.Iter = 0

    # lazy init
    def initMV(self, w):
        if (False == self.isInited):
            for i in range(len(w)):
                self.m.append(np.zeros(w[i].shape, dtype=self.dataType))
                self.v.append(np.zeros(w[i].shape, dtype=self.dataType))
            # self.m_w = np.zeros(shapeW, dtype=self.dataType)
            # self.v_w = np.zeros(shapeW, dtype=self.dataType)
            # self.m_b = np.zeros(shapeB, dtype=self.dataType)
            # self.v_b = np.zeros(shapeB, dtype=self.dataType)
            self.isInited = True

    def getUpdWeights(self, w, dw, lr):
        self.initMV(w)

        t = self.Iter + 1
        wNew = []
        for i in range(len(w)):
            wi, self.m[i],self.v[i] = self.OptimzAdam(w[i], dw[i], self.m[i], self.v[i], lr, t)
            wNew.append(wi)

        # 转为元组输出
        return tuple(wNew)

    def OptimzAdam(self, x, dx, m, v, lr, t):
        m = self.beta1 * m + (1 - self.beta1) * dx
        mt = m / (1 - self.beta1 ** t)
        v = self.beta2 * v + (1 - self.beta2) * (dx ** 2)
        vt = v / (1 - self.beta2 ** t)
        x += - lr * mt / (np.sqrt(vt) + self.eps)

        return x, m, v

# 自适应矩估计优化类
# TODO 固定接受w,b需要改为更灵活的形式如 adagrad传入元组的方式
class AdamOptimizer_succ(object):
    def __init__(self, beta1, beta2, eps, dataType):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.dataType = dataType
        self.isInited = False
        self.m_w = []
        self.v_w = []
        self.m_b = []
        self.v_b = []
        self.Iter = 0

    # lazy init
    def initMV(self, shapeW, shapeB):
        if (False == self.isInited):
            self.m_w = np.zeros(shapeW, dtype=self.dataType)
            self.v_w = np.zeros(shapeW, dtype=self.dataType)
            self.m_b = np.zeros(shapeB, dtype=self.dataType)
            self.v_b = np.zeros(shapeB, dtype=self.dataType)
            self.isInited = True

    def getUpdWeights(self, w, dw, b, db, lr):
        self.initMV(w.shape, b.shape)

        t = self.Iter + 1
        wNew, self.m_w, self.v_w = self.OptimzAdam(w, dw, self.m_w, self.v_w, lr, t)
        bNew, self.m_b, self.v_b = self.OptimzAdam(b, db, self.m_b, self.v_b, lr, t)
        self.Iter += 1
        return wNew, bNew

    def OptimzAdam(self, x, dx, m, v, lr, t):
        # beta1 = self.beta1
        # beta2 = self.beta2
        m = self.beta1 * m + (1 - self.beta1) * dx
        mt = m / (1 - self.beta1 ** t)
        v = self.beta2 * v + (1 - self.beta2) * (dx ** 2)
        vt = v / (1 - self.beta2 ** t)
        x += - lr * mt / (np.sqrt(vt) + self.eps)

        return x, m, v
