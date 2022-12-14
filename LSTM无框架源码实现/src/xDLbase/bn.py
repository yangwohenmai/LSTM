"""
Created: May 2018
@author: JerryX
Find more : https://www.zhihu.com/people/xu-jerry-82
"""

import numpy as np
import time

# import logging.config

# create logger
# import os
# exec_abs = os.getcwd()
# log_conf = exec_abs + '/config/logging.conf'
# logging.config.fileConfig(log_conf)
# logger = logging.getLogger('bn')


# batch normalization layer
# with optimizer adopter
# gamma和beta由传入x的维度来确定
class BNLayer(object):

    def __init__(self,LName,eps,miniBatchesSize,channel,i_size,activator,optimizerCls,optmParams, dataType):
        self.name = LName
        self.miniBatchesSize = miniBatchesSize
        self.i_size = i_size
        self.channel = channel
        self.activator = activator
        self.eps = eps
        self.optimizerObj = optimizerCls(optmParams, dataType)
        self.dataType = dataType
        self.beta = []
        self.gamma = []
        self.mu_accum = []
        self.var_accum = []
        self.isInited = False
        self.decay_accum = 0.95  # 滑动平均衰减率
        self.cache = ()
        self.out = []
        self.deltaPrev = []  # 上一层激活后的误差输出
        self.deltaOri = []  # 本层原始误差输出
        #self.lazy_init_flg = False # 初始化标志  abandon

    # lazy init
    def initParam(self, x):
        if (False == self.isInited):
            if len(x.shape) == 2:
                N,D = x.shape
                paramShape=(D)
            if len(x.shape) == 4:
                N,C,W,H = x.shape
                # paramShape=(N*W*H,C)
                paramShape=(C)
            # 初始化滑动平均统计量和训练参数
            self.initParamHelper(paramShape)
            self.isInited = True

    def initParamHelper(self,paramShape):
        self.beta = np.zeros((paramShape),self.dataType)
        self.gamma = np.ones((paramShape),self.dataType)
        self.mu_accum = np.zeros((paramShape),self.dataType)
        self.var_accum = np.ones((paramShape),self.dataType)

    # 滑动平均衰减累计mini-batch的mean和variance
    def mov_avg_accum(self,mu,var):
        decay = self.decay_accum

        #计算滑动平均
        self.mu_accum = decay * self.mu_accum + (1-decay) * mu
        self.var_accum = decay * self.var_accum + (1-decay) * var

    def inference(self,x):
        # oriOut = self.bnForward_inf(x,self.gamma,self.beta,self.eps)
        oriOut = self.bnForward_inf(x,self.eps)
        self.out = self.activator.activate(oriOut)
        return self.out

    def fp(self,x):
        # oriOut,self.cache = self.bnForward_tr(x,self.gamma,self.beta,self.eps)
        oriOut,self.cache = self.bnForward_tr(x,self.eps)
        self.out = self.activator.activate(oriOut)
        return self.out

    # for training
    # def bnForward_tr(self, xOri, gamma, beta, eps):
    def bnForward_tr(self, xOri, eps):

      assert len(xOri.shape) in (2, 4)
      self.initParam(xOri)
      if len(xOri.shape) == 4:
          N,C,H,W = xOri.shape
          x = xOri.transpose(0, 2, 3, 1).reshape(N * H * W, C)
      elif len(xOri.shape) == 2:
          N,D = xOri.shape
          x = xOri

      mu = np.mean(x, axis=0)
      xmu = x - mu
      var = np.mean(xmu **2, axis = 0)

      #累计每个批次的 mean和var
      self.mov_avg_accum(mu, var)

      ivar = 1./np.sqrt(var + eps)
      xhat = xmu * ivar
      # out = gamma*xhat + beta
      out = self.gamma*xhat + self.beta

      if len(xOri.shape) == 4:
          outOri = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
      else:
          outOri = out

      # cache = (xhat,gamma,ivar)
      cache = (xhat,ivar)

      return outOri, cache

    # 优化，不使用分步骤方式
    def bnBackward(self,doutOri, cache):
      assert len(doutOri.shape) in (2, 4)

      if len(doutOri.shape) == 4:
          N, C, H, W = doutOri.shape
          dout = doutOri.transpose(0, 2, 3, 1).reshape(N * H * W, C)
      else:
          dout = doutOri


      st = time.time()
      # logger.debug('bnBk start')

      # xhat,gamma,ivar = cache
      xhat,ivar = cache

      Nt = dout.shape[0]

      dbeta = np.sum(dout, axis=0)
      dgamma = np.sum(dout*xhat, axis=0)

      # dxhat = dout * gamma
      dxhat = dout * self.gamma
      # dx = 1./Nt* ivar * (Nt*dxhat-np.sum(dxhat, axis=0) - xhat*np.sum(dxhat*xhat,axis=0))
      dx = 1./Nt* ivar * (Nt*dxhat-np.sum(dxhat, axis=0) - ivar*xhat*np.sum(dxhat*xhat,axis=0))

      # logger.debug('bnBk end, %s s', (time.time()-st) )
      if len(doutOri.shape) == 4:
        dxOri = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
      else:
        dxOri = dx

      return dxOri, dgamma, dbeta


    # for predict, Unbiased Estimation
    def bnForward_inf(self, xOri, eps):
      assert len(xOri.shape) in (2, 4)

      if len(xOri.shape) == 4:
          N, C, H, W = xOri.shape
          x = xOri.transpose(0, 2, 3, 1).reshape(N * H * W, C)
      elif (xOri.shape == 2):
          N, D = xOri.shape
          x = xOri

      M = self.miniBatchesSize


      var = M / (M - 1) * self.var_accum
      tx = (x - self.mu_accum) /np.sqrt(var + eps)
      out = self.gamma * tx + self.beta

      if len(xOri.shape) == 4:
          outOri = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
      else:
          outOri = out

      return outOri

    def bp(self,input,delta,lrt):
        self.deltaOri = self.activator.bp(delta, self.out)

        self.deltaPrev, dgamma, dbeta = self.bnBackward(self.deltaOri, self.cache)

        weight=(self.gamma,self.beta)
        dweight = (dgamma,dbeta)
        self.optimizerObj.getUpdWeights(weight,dweight, lrt)

        return self.deltaPrev


class BNLayerPerPixer(object):

    def __init__(self,LName,eps,miniBatchesSize,channel,i_size,activator,optimizerCls,optmParams, dataType):
        self.name = LName
        self.miniBatchesSize = miniBatchesSize
        self.i_size = i_size
        self.channel = channel
        self.activator = activator
        self.eps = eps
        self.optimizerObj = optimizerCls(optmParams, dataType)
        self.dataType = dataType
        self.beta = np.zeros((channel,i_size,i_size),dataType)
        self.gamma = np.ones((channel,i_size,i_size),dataType)
        self.mu_accum = np.zeros((channel,i_size,i_size),dataType)
        self.var_accum = np.ones((channel,i_size,i_size),dataType)
        self.decay_accum = 0.95  # 滑动平均衰减率
        self.cache = ()
        self.out = []
        self.deltaPrev = []  # 上一层激活后的误差输出
        self.deltaOri = []  # 本层原始误差输出
        #self.lazy_init_flg = False # 初始化标志  abandon

    # 滑动平均衰减累计mini-batch的mean和variance
    def mov_avg_accum(self,mu,var):
        decay = self.decay_accum

        #计算滑动平均
        self.mu_accum = decay * self.mu_accum + (1-decay) * mu
        self.var_accum = decay * self.var_accum + (1-decay) * var

    def inference(self,x):
        oriOut = self.bnForward_inf(x,self.gamma,self.beta,self.eps)
        self.out = self.activator.activate(oriOut)
        return self.out

    def fp(self,x):
        oriOut,self.cache = self.bnForward_tr(x,self.gamma,self.beta,self.eps)
        self.out = self.activator.activate(oriOut)
        return self.out

    # for training
    def bnForward_tr(self, x, gamma, beta, eps):

      mu = np.mean(x, axis=0)
      xmu = x - mu
      var = np.mean(xmu **2, axis = 0)

      #累计每个批次的 mean和var
      self.mov_avg_accum(mu, var)

      ivar = 1./np.sqrt(var + eps)
      xhat = xmu * ivar
      out = gamma*xhat + beta

      cache = (xhat,gamma,ivar)

      return out, cache

    # 优化，不使用分步骤方式
    def bnBackward(self,dout, cache):

      st = time.time()
      # logger.debug('bnBk start')

      xhat,gamma,ivar = cache

      N = dout.shape[0]

      dbeta = np.sum(dout, axis=0)
      dgamma = np.sum(dout*xhat, axis=0)

      dxhat = dout * gamma
      dx = 1./N* ivar * (N*dxhat-np.sum(dxhat, axis=0) - xhat*np.sum(dxhat*xhat,axis=0))

      # logger.debug('bnBk end, %s s', (time.time()-st) )

      return dx, dgamma, dbeta


    # for predict, Unbiased Estimation
    def bnForward_inf(self, x, gamma, beta, eps):

      N = self.miniBatchesSize

      var = N / (N - 1) * self.var_accum
      tx = (x - self.mu_accum) /np.sqrt(var + eps)
      out = gamma * tx + beta

      return out

    def bp(self,input,delta,lrt):
        self.deltaOri = self.activator.bp(delta, self.out)

        self.deltaPrev, dgamma, dbeta = self.bnBackward(self.deltaOri, self.cache)

        weight=(self.gamma,self.beta)
        dweight = (dgamma,dbeta)
        self.optimizerObj.getUpdWeights(weight,dweight, lrt)

        return self.deltaPrev

