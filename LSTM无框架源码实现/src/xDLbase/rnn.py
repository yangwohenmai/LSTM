"""
Some different RNN Model
"""

import numpy as np
import operator as op
import numba

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)
from xDLUtils import Tools
from activators import ReLU

# Rnn类
class RnnLayer(object):
    # N,H,L和优化器在初始化时定义
    # T作为X的一个维度传进来
    # tanh和sigmoid的前反向传播在类内部定义。
    def __init__(self, LName, miniBatchesSize, nodesNum, layersNum,
                 optimizerCls, optmParams, dropoutRRate, dataType,  init_rng):
        # 初始化超参数
        self.name = LName
        self.miniBatchesSize = miniBatchesSize
        self.nodesNum = nodesNum
        self.layersNum = layersNum
        self.dataType = dataType
        self.init_rng = init_rng
        self.isInited = False  # 初始化标志
        # dropout 的保留率
        self.dropoutRRate = dropoutRRate
        self.dropoutMask = []

        self.out = []
        self.optimizerObjs = [optimizerCls(optmParams, dataType) for i in range(layersNum)]

        # 初始化w,u,b 和对应偏置,维度，层次和节点个数传参进去。但是没有T，所以不能创建参数
        # 返回的是一个组合结构,按层次（数组）划分的U、W，字典
        # 改为放在首batch X传入时lazy init
        self.rnnParams = []

        # 保存各层中间产出的 st和f(st)，用于前向和反向传播
        # 不需要，已经在前反向传播中保留
        self.deltaPrev = []  # 上一层激活后的误差输出

    def _initNnWeight(self, D, H, layersNum, dataType):

        # 层次
        rnnParams = []
        for layer in range(layersNum):

            Wh = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, H)).astype(dataType)
            if (0 == layer):
                Wx = np.random.uniform(-1 * self.init_rng, self.init_rng, (D, H)).astype(dataType)
            else:
                Wx = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, H)).astype(dataType)
            b = np.zeros(H, dataType)
            rnnParams.append({'Wx': Wx, 'Wh': Wh, 'b': b})

        self.rnnParams = rnnParams

    def _initNnWeightOrthogonal(self, D, H, layersNum, dataType):

        # 层次
        rnnParams = []
        for layer in range(layersNum):

            # Wh = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, H)).astype(dataType)
            Wh = Tools.initOrthogonal( (H, H),self.init_rng, dataType)

            DH = D if 0 == layer else H
            Wx = Tools.initOrthogonal( (DH, H),self.init_rng, dataType)
            b = np.zeros(H, dataType)
            rnnParams.append({'Wx': Wx, 'Wh': Wh, 'b': b})

        self.rnnParams = rnnParams


    # 训练时前向传播
    def fp(self, input):
        out_tmp = self.inference(input)
        self.out, self.dropoutMask = Tools.dropout4rnn(out_tmp, self.dropoutRRate)
        return self.out

    # 预测时前向传播,激活后再输出
    # input: batch x seqNum, 32*10
    def inference(self, x):
        N, T, D = x.shape
        H = self.nodesNum
        L = self.layersNum
        # lazy init
        if (False == self.isInited):
            #self._initNnWeight(D, H, L, self.dataType)
            self._initNnWeightOrthogonal(D, H, L, self.dataType)
            self.isInited = True

        # 缓存已经存入rnnParams里了,此处只需要返回输出结果(N,T,H)
        h = self.rnn_forward(x)
        # N进 v 1出 模型，只保留时序最后的一项输出
        # self.out = h[:,-1,:]
        # 全部输出,未用到的部分梯度为0
        self.out = h
        return self.out

    # 反向传播方法(误差和权参)
    # TODO 实现反向传播逻辑，先按照时间，再按照层次，再更新Wx/Wf/b/V/bv 及偏置的反向传播梯度
    def bp(self, input, delta_ori, lrt):

        if self.dropoutRRate == 1:
            delta = delta_ori
        else:
            delta = delta_ori * self.dropoutMask

        # dw是一个数组，对应结构的多层，每层的dw,dh,db,dh0表示需要参数梯度
        N, T, D = input.shape
        H = delta.shape[1]
        # 只有最后一个T填delta，其余的dh梯度设置为0
        dh = np.zeros((N, T, H), self.dataType)
        # dh[:,-1,:] = delta
        dh = delta
        dx, dweight = self.rnn_backward(dh)

        # 根据梯度更新参数
        self.bpWeights(dweight, lrt)

        return dx

    # 计算反向传播权重梯度w,b
    def bpWeights(self, dw, lrt):

        L = self.layersNum
        # for l in range(L - 1, -1, -1):
        for l in range(L):
            w = (self.rnnParams[l]['Wx'], self.rnnParams[l]['Wh'], self.rnnParams[l]['b'])
            # 此处不赋值也可以，因为是按引用传参
            # self.rnnParams[l]['Wx'], self.rnnParams[l]['Wh'], self.rnnParams[l]['b'] = self.optimizerObjs[l].getUpdWeights(w,dw[L-1-l],lrt)
            self.optimizerObjs[l].getUpdWeights(w, dw[L - 1 - l], lrt)

    def rnn_forward(self, x):
        """
        Run a vanilla RNN forward on an entire sequence of data. We assume an input
        sequence composed of T vectors, each of dimension D. The RNN uses a hidden
        size of H, and we work over a minibatch containing N sequences. After running
        the RNN forward, we return the hidden states for all timesteps.

        Inputs:
        - x: Input data for the entire timeseries, of shape (N, T, D).
        - h0: Initial hidden state, of shape (N, H)
        - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        - b: Biases of shape (H,)

        Returns a tuple of:
        - h: Hidden states for the entire timeseries, of shape (N, T, H).
        - cache: Values needed in the backward pass
        """

        h, cache = None, None
        ##############################################################################
        # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
        # input data. You should use the rnn_step_forward function that you defined  #
        # above. You can use a for loop to help compute the forward pass.            #
        ##############################################################################

        N, T, D = x.shape
        L = self.layersNum
        H = self.rnnParams[0]['b'].shape[0]
        xh = x
        for layer in range(L):

            h = np.zeros((N, T, H))
            h0 = np.zeros((N, H))
            cache = []
            for t in range(T):
                h[:, t, :], tmp_cache = self.rnn_step_forward(xh[:, t, :],
                                                              h[:, t - 1, :] if t > 0 else h0,
                                                              self.rnnParams[layer]['Wx'], self.rnnParams[layer]['Wh'],
                                                              self.rnnParams[layer]['b'])
                cache.append(tmp_cache)
            xh = h  # 之后以h作为xh作为跨层输入
            ##############################################################################
            #                               END OF YOUR CODE                             #
            ##############################################################################
            self.rnnParams[layer]['h'] = h
            self.rnnParams[layer]['cache'] = cache

        return h  # 返回最后一层作为输出

    def rnn_backward(self, dh):
        """
        Compute the backward pass for a vanilla RNN over an entire sequence of data.

        Inputs:
        - dh: Upstream gradients of all hidden states, of shape (N, T, H).

        NOTE: 'dh' contains the upstream gradients produced by the
        individual loss functions at each timestep, *not* the gradients
        being passed between timesteps (which you'll have to compute yourself
        by calling rnn_step_backward in a loop).

        Returns a tuple of:
        - dx: Gradient of inputs, of shape (N, T, D)
        - dh0: Gradient of initial hidden state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
        - db: Gradient of biases, of shape (H,)
        """
        dx, dh0, dWx, dWh, db = None, None, None, None, None
        ##############################################################################
        # TODO: Implement the backward pass for a vanilla RNN running an entire      #
        # sequence of data. You should use the rnn_step_backward function that you   #
        # defined above. You can use a for loop to help compute the backward pass.   #
        ##############################################################################
        N, T, H = dh.shape
        x, _, _, _, _ = self.rnnParams[0]['cache'][0]
        D = x.shape[1]

        # 初始化最上一层误差

        dh_prevl = dh
        # 保存各层dwh,dwx,和db
        dweights = []
        # 逐层倒推
        for layer in range(self.layersNum - 1, -1, -1):
            # 得到前向传播保存的cache数组
            cache = self.rnnParams[layer]['cache']

            DH = D if layer == 0 else H
            dx = np.zeros((N, T, DH))
            dWx = np.zeros((DH, H))
            dWh = np.zeros((H, H))
            db = np.zeros(H)
            dprev_h_t = np.zeros((N, H))
            # 倒序遍历
            for t in range(T - 1, -1, -1):
                dx[:, t, :], dprev_h_t, dWx_t, dWh_t, db_t = self.rnn_step_backward(dh_prevl[:, t, :] + dprev_h_t,
                                                                                    cache[t])
                dWx += dWx_t
                dWh += dWh_t
                db += db_t

            # 本层得出的dx，作为下一层的prev_l
            dh_prevl = dx

            dweight = (dWx, dWh, db)
            dweights.append(dweight)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        # 返回x误差和各层参数误差
        return dx, dweights

    def rnn_step_forward(self, x, prev_h, Wx, Wh, b):
        """
        Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
        activation function.

        The input data has dimension D, the hidden state has dimension H, and we use
        a minibatch size of N.

        Inputs:
        - x: Input data for this timestep, of shape (N, D).
        - prev_h: Hidden state from previous timestep, of shape (N, H)
        - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        - b: Biases of shape (H,)

        Returns a tuple of:
        - next_h: Next hidden state, of shape (N, H)
        - cache: Tuple of values needed for the backward pass.
        """

        next_h, cache = None, None
        ##############################################################################
        # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
        # hidden state and any values you need for the backward pass in the next_h   #
        # and cache variables respectively.                                          #
        ##############################################################################
        z = Tools.matmul(x, Wx) + Tools.matmul(prev_h, Wh) + b

        next_h = np.tanh(z)

        dtanh = 1. - next_h * next_h
        cache = (x, prev_h, Wx, Wh, dtanh)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return next_h, cache

    def rnn_step_backward(self, dnext_h, cache):
        """
        Backward pass for a single timestep of a vanilla RNN.

        Inputs:
        - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
        - cache: Cache object from the forward pass

        Returns a tuple of:
        - dx: Gradients of input data, of shape (N, D)
        - dprev_h: Gradients of previous hidden state, of shape (N, H)
        - dWx: Gradients of input-to-hidden weights, of shape (D, H)
        - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
        - db: Gradients of bias vector, of shape (H,)
        """
        dx, dprev_h, dWx, dWh, db = None, None, None, None, None
        ##############################################################################
        # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
        #                                                                            #
        # HINT: For the tanh function, you can compute the local derivative in terms #
        # of the output value from tanh.                                             #
        ##############################################################################
        x, prev_h, Wx, Wh, dtanh = cache
        dz = dnext_h * dtanh
        dx = Tools.matmul(dz, Wx.T)
        dprev_h = Tools.matmul(dz, Wh.T)
        dWx = Tools.matmul(x.T, dz)
        dWh = Tools.matmul(prev_h.T, dz)
        db = np.sum(dz, axis=0)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return dx, dprev_h, dWx, dWh, db


# LSTM 类
class LSTMLayer(object):
    # N,H,L和优化器在初始化时定义
    # T作为X的一个维度传进来
    # tanh和sigmoid的前反向传播在类内部定义。
    def __init__(self, LName, miniBatchesSize, nodesNum, layersNum,
                 optimizerCls, optmParams, dropoutRRate, dataType, init_rng):
        # 初始化超参数
        self.name = LName
        # 批次数
        self.miniBatchesSize = miniBatchesSize
        # 每层神经元数量
        self.nodesNum = nodesNum
        # 模型网络层数
        self.layersNum = layersNum
        self.dataType = dataType
        # 权重矩阵初始值
        self.init_rng = init_rng
        self.isInited = False  # 初始化标志

        # dropout 的保留率
        self.dropoutRRate = dropoutRRate
        self.dropoutMask = []

        self.out = []
        # 三层LSTM每层定义化优化器
        self.optimizerObjs = [optimizerCls(optmParams, dataType) for i in range(layersNum)]
        # 初始化w,u,b 和对应偏置,维度，层次和节点个数传参进去。但是没有T，所以不能创建参数
        # 返回的是一个组合结构,按层次（数组）划分的U、W，字典
        # 改为放在首batch X传入时lazy init
        self.lstmParams = []

        # 保存各层中间产出的 st和f(st)，用于前向和反向传播
        # 不需要，已经在前反向传播中保留
        self.deltaPrev = []  # 上一层激活后的误差输出

    def _initNnWeight(self, D, H, layersNum, dataType):

        # 层次
        lstmParams = []
        for layer in range(layersNum):
            Wh = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 4 * H)).astype(dataType)
            if (0 == layer):
                Wx = np.random.uniform(-1 * self.init_rng, self.init_rng, (D, 4 * H)).astype(dataType)
            else:
                Wx = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 4 * H)).astype(dataType)
            b = np.zeros(4 * H, dataType)

            lstmParams.append({'Wx': Wx, 'Wh': Wh, 'b': b})
        self.lstmParams = lstmParams

    # 初始化网络LSTM层的权重矩阵，参数设定有3个LSTM层，每层30个隐藏神经元
    # layersNum=3 设置3层LSTM
    # Dime=2 特征值有两个
    # H=30 L每层STM中设置神经元节点30个
    def _initNnWeightOrthogonal(self, Dime , H, layersNum, dataType):
        # 保存配置好的每层LSTM的权重矩阵参数到self.lstmParams中
        lstmParams = []
        # 给每层网络配置[Wx Wh b]三个权重矩阵，共3层LSTM
        for layer in range(layersNum):
            # Wh = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 4 * H)).astype(dataType)
            # Wh指上一时间步状态h到f,i,g,o仿射变换的权重矩阵
            # 设置Wh.shape(H, 4*H)=(30, 120),这里将f,i,g,o四个矩阵进行合并，方便计算
            Wh =  Tools.initOrthogonal( (H, 4*H),self.init_rng, dataType)
            # 设置Wx矩阵shape(D_or_H, 4*H)：如果是第0层则D_or_H=Dime，即矩阵的维度等于输入特征的维度2；否则D_or_H=H，即向后传播的维度30
            D_or_H = Dime if 0 == layer else H
            # Wx = np.random.uniform(-1 * self.init_rng, self.init_rng, (D_or_H, 4 * H)).astype(dataType)
            # Wx指当前时间步输入x到f,i,g,o仿射变换的权重矩阵
            # 设置不同形状的Wx矩阵，第一层的Wx接受输入参数，和后续层的Wx矩阵shape不同。同上将f,i,g,o四个矩阵进行合并
            # 1.Wx(2, 120)，首层接收到的输入x为(32,1,2)
            # 2.Wx(30, 120),后续接收到的输入x为(32,1,30)
            # 3.Wx(30, 120)
            Wx =  Tools.initOrthogonal( (D_or_H, 4*H),self.init_rng, dataType)
            # b.shape(120,)偏置项都是120
            b = np.zeros(4 * H, dataType)
            lstmParams.append({'Wx': Wx, 'Wh': Wh, 'b': b})
        # 存储3个LSTM层的权重矩阵参数[Wh Wx b]
        self.lstmParams = lstmParams

    # 预测时前向传播
    def fp(self, input):
        # 输入x的格式input(32, 1, 2)，2是特征维度；输出隐藏层的格式out_tmp,h=(32, 1, 30)，30是隐藏层节点
        out_tmp = self.inference(input)
        self.out, self.dropoutMask = Tools.dropout4rnn(out_tmp, self.dropoutRRate)
        return self.out

    def inference(self, x):
        # 输入x.shape(32, 1, 2)
        Sample, Timestep, Dim = x.shape
        # 定义的隐藏层神经元节点数HiddenNode=30
        HiddenNode = self.nodesNum
        # 定义的模型中需要配置的LSTM层数，共3层LSTM
        Layer = self.layersNum
        # lazy init
        if (False == self.isInited):
            # self._initNnWeight(Dim, HiddenNode, Lay, self.dataType)
            # 配置模型中3层LSTM的每一层的初始权重矩阵，每个LSTM层包含[Wh Wx b]三个矩阵，首层Wx矩阵形状略有不同，因为接收的是输入X的shape
            self._initNnWeightOrthogonal(Dim, HiddenNode, Layer, self.dataType)
            self.isInited = True

        # 上一步配置好3层LSTM网络层，已经存入缓存rnnParams里,此处返回的是当前网络层输出的状态h，h.shape=(Sample,Timestep,H)
        h = self.lstm_forward(x)
        
        # self.out = h[:,-1,:]
        # 记录3层LSTM最后一次输出的h(32, 1, 30)
        self.out = h
        return self.out

    # 反向传播方法(误差和权参)
    # TODO 实现反向传播逻辑，先按照时间，再按照层次，再更新Wx/Wf/b/V/bv 及偏置的反向传播梯度
    def bp(self, input, delta_ori, lrt):
        # 误差矩阵是否需要dropout
        if self.dropoutRRate == 1:
            delta = delta_ori
        else:
            delta = delta_ori * self.dropoutMask

        # dw是一个数组，对应结构的多层，每层的dw,dh,db,dh0表示需要参数梯度(32, 1, 2)
        N, T, D = input.shape
        #(32, 1, 30)
        H = delta.shape[1]
        #(32, 1, 1)
        dh = np.zeros((N, T, H), self.dataType)
        # 只有最后一个T填delta，其余的dh梯度设置为0
        # dh[:,-1,:] = delta
        #(32, 1, 30)
        dh = delta
        dx, dweight = self.lstm_backward(dh)
        # 通过优化器对反向传播进行梯度优化
        self.bpWeights(dweight, lrt)

        return dx

    # 计算反向传播权重梯度w,b
    def bpWeights(self, dweight, lrt):

        L = self.layersNum
        for l in range(L):
            w = (self.lstmParams[l]['Wx'], self.lstmParams[l]['Wh'], self.lstmParams[l]['b'])
            # self.lstmParams[l]['Wx'], self.lstmParams[l]['Wh'], self.lstmParams[l]['b'] = self.optimizerObjs[l].getUpdWeights(w, dweight[L-1-l], lrt)
            # 元组按引用对象传递，值在方法内部已被更新
            # 将两个元组代入优化器，求出梯度方向，根据梯度更新参数，更新了self.lstmParams[i][Wx,Wh,b]的权重矩阵
            self.optimizerObjs[l].getUpdWeights(w, dweight[L - 1 - l], lrt)
            # self.optimizerObjs[l].getUpdWeights(w, dweight[l], lrt)
    

    #############################################################################
    # 多层LSTM 多时间步 前向传播算法，此处需注意xh值在不同层对应的参数意义
    # Input
    # - x: 训练集输入数据 (N, T, D)
    # - xh: 每层LSTM间传递的参数，首层xh为训练集输入x，后续xh为上层LSTM每个时间步的输出状态h(N, T, D)
    # - h0: 首层LSTM传入的h状态 shape (N, H)
    # - c0: 首层LSTM传入的c状态  shape(N, H)
    # - h: 后续层LSTM传入的h状态 shape(N, T, H)
    # - c: 后续层LSTM传入的c状态 shape(N, T, H)
    # - Wx: x到f,i,g,o的权重矩阵 shape(D, 4H)，首层(2,120)，2,3层(30,120)
    # - Wh: h到f,i,g,o的权重矩阵 shape(H, 4H)，3层都是(30,120)
    # - b: 偏置项 shape(4H,)
    # Returns a tuple of:
    # - h: 每个时间步输出的状态 shape(N, T, H)
    # - cache: 每个时间步产生的关于f,i,g,o门的中间参数
    #############################################################################
    def lstm_forward(self, x):
        h, cache = None, None
        # x.shape(32, 1, 2)
        N, T, D = x.shape
        # 根据权重矩阵中偏置项b的shape来获取Hidden层的节点数
        H = int(self.lstmParams[0]['b'].shape[0] / 4)  # 取整
        # 首次计算时只存在输入x，不存在h，所以传入xh值为输入x,xh表示上个时间步的状态h
        # 1. 首次输入xh=x.shape(32, 1, 2)
        xh = x
        '''
        当前循环负责：3个LSTM层之间的参数传递，每层之间传参时，h,c,cache全都重新初始化；
        其中xh表示每个LSTM层的输入，可以为x/h，h[]，c[]记录每个时间步生成的两个状态，cache存储f,i,g,o门产生的中间参数；
        首层的xh为输入的训练数据x，后续层的xh为上一层LSTM在每个时间步所生成的状态h[,,]，二者shape可能会有差异
        '''
        for layer in range(self.layersNum):
            # 每个LSTM层首次计算时要初始化当前h和c为0矩阵，类似reset_states作用，(N, T, H)=(32, 1, 30)
            h = np.zeros((N, T, H))
            c = np.zeros((N, T, H))
            # h0,c0作为本层首个时间步的初始化参数，(N, H)=shape(32, 30)
            h0 = np.zeros((N, H))
            c0 = np.zeros((N, H))
            cache = []
            '''
            当前循环负责：对N个sample中 每一个sample内部的 每个timesteps间的参数传递。每轮循环将一个batch中包含的N个sample同时传入，并行计算，最后也会同时输出N组预测结果；
            这里的T对应的是每个sample中所包含的时间步timesteps个数，程序按照每个时间步来进行循环前向传播计算；
            每次计算时将上一时间步输出的状态h[:,t-1,:], c[:,t-1,:]作为当前时间步的输入；当前时间步输出的状态存入h[:,t,:],c[:,t,:]中，作为后续的输入
            '''
            for t in range(T):
                # (h0,c0).shape = (h[:,t-1,:],c[:,t-1,:]).shape = (32, 30)
                # 每轮出参的h,cshape相同,h(32, 1, 30),c(32, 1, 30)，此例子时间步T为1，所以只进行一轮循环计算 
                h[:, t, :], c[:, t, :], tmp_cache = self.lstm_step_forward(xh[:, t, :], 
                                                                            h[:, t - 1, :] if t > 0 else h0,
                                                                            c[:, t - 1, :] if t > 0 else c0,
                                                                            self.lstmParams[layer]['Wx'], self.lstmParams[layer]['Wh'], self.lstmParams[layer]['b'])
                cache.append(tmp_cache)
            # 计算完当前LSTM层所有时间步，将每个时间步生成的h,c集合代入下一层的LSTM进行跨层运算
            # 2.xh(32, 1, 30) xh为上个LSTM层每个时间步的状态集合h(32, 1, 30)
            # 3.xh(32, 1, 30) xh为上个LSTM层每个时间步的状态集合h(32, 1, 30)
            xh = h
            ##############################################################################
            #                               END OF YOUR CODE                             #
            ##############################################################################
            self.lstmParams[layer]['h'] = h
            self.lstmParams[layer]['c'] = c
            self.lstmParams[layer]['cache'] = cache
        # 最终将最后一个LSTM层的每个时间步生成的h值返回，作为新输入传给FNN全连接层
        return h

    #############################################################################
    # TODO: 两个for循环实现：N层堆叠LSTM模型，每个sample包含T个时间步，的backward计算
    # Inputs:
    # - dh: 全连接层反向传入堆叠LSTM中的，最后一个LSTM层的，误差矩阵(N, T, H)
    # - cache: 前向传播过程中产生的中间变量，如f,i,g,o,H,C等
    # Returns a tuple of:
    # - dx: 存储 包含T个时间步的误差矩阵(N, T, D)
    # - dWx: 存储 输入x<-控制门 权重矩阵的梯度(D, 4H)
    # - dWh: 存储 状态h<-控制门 权重矩阵的梯度(H, 4H)
    # - db: 偏置项的梯度(4H,)
    #############################################################################
    def lstm_backward(self, dh):
        dx, dh0, dWx, dWh, db = None, None, None, None, None
        #(32, 1, 30)
        N, T, H = dh.shape
        # 获取第一层输入的shape，第一层输入矩阵和隐藏层前向传播矩阵shape不同，lstmParams包含Wx Wh b h c cache
        x, _, _, _, _, _, _, _, _, _ = self.lstmParams[0]['cache'][0]
        #(32, 2)
        D = x.shape[1]
        # dh是全连接层传入的误差矩阵(32, 1, 30)
        dh_prevl = dh
        # 保存各层dwh,dwx,和db
        dweights = []

        for layer in range(self.layersNum - 1, -1, -1):
            # 得到前向传播保存的cache数组
            cache = self.lstmParams[layer]['cache']
            # 如果是第一层网络layer==0，则矩阵shape(DH=D)要与输入矩阵相同；
            # 如果是中间隐藏层矩阵，则shape(DH=H)替换成hidden神经节点数
            DH = D if layer == 0 else H
            # dWx(32, 1, 30)
            dx = np.zeros((N, T, DH))
            # dWx(30, 120)
            dWx = np.zeros((DH, 4 * H))
            # dWh(30, 120)
            dWh = np.zeros((H, 4 * H))
            # db(120,)
            db = np.zeros((4 * H))
            # dprev_h(32, 30)
            dprev_h = np.zeros((N, H))
            # dprev_c(32, 30),上述所有初始化的零矩阵，只有dprev_c，dprev_h是作为参数传入，其他主要作为接收输出用
            dprev_c = np.zeros((N, H))
            # T是一个sample中包含的时间timestep数，对全连接层传入的误差矩阵dh_prevl，按照每个时间步timesteps循环(倒序)，进行反向传播计算
            # 每步计算后将对应的状态存入dx对应的时间步位置，作为下个反向传播层/LSTM层的输入状态dh_prevl，cache是前向传播时记录的中间参数，用于后续求导
            # 将dh_prevl[:,t,:]+dprev_h对应反向传播时H_t位置的输入参数，dh_prevl[:,t,:]存储的是上个LSTM层反向传播中，对应原输入x位置的T个时间步的误差结果。在本层LSTM按时间步计算时，每次输入dh_prevl中一个时间步T对应的误差，同时再加上本层上一个时间步输出的误差dprev_h
            for t in range(T - 1, -1, -1):
                dx[:, t, :], dprev_h, dprev_c, dWx_t, dWh_t, db_t = self.lstm_step_backward(dh_prevl[:, t, :] + dprev_h, dprev_c, cache[t])  # 注意此处的叠加
                dWx += dWx_t
                dWh += dWh_t
                db += db_t

            # 本层求得的n维timestep反向传播误差矩阵dx，作为下一层的输入dh_prevl继续传递；其他dprev_h,dprev_c等变量在当前sample所有时间步计算结束时舍弃。
            # 因为LSTM堆叠时，前向传播除了首层LSTM的输入是训练数据x，后续层的输入都是将上层的输出状态H作为下层的输入x，所以反向传播只用回传dx即可
            dh_prevl = dx
            # 存入dweight后续用梯度下降更新矩阵参数
            dweight = (dWx, dWh, db)
            dweights.append(dweight)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        # 返回dx误差和各层参数误差
        return dx, dweights


    #############################################################################
    # LSTM单元控制门内部算法实现
    # Inputs:
    # - x: 输入为训练数据x，或上一层LSTM生成的每个时间步的状态参数h集合, shape(N, D)
    # - prev_h: 本层LSTM上个时间步生成的状态h, shape(N, H)
    # - prev_c: 本层LSTM上个时间步生成的状态c, shape(N, H)
    # - Wx: 输入xh-figo门权重矩阵, shape(D, 4H)
    # - Wh: 隐藏状态h-figo门权重矩阵, shape(H, 4H)
    # - b: 偏置项, shape(4H,)
    # Returns a tuple of:
    # - next_h: 当前时间步计算的h.shape(N, H)，传给下个时间步
    # - next_c: 当前时间步计算的c.shape(N, H)，传给下个时间步
    # - cache: 反向传播需要用到的f,i,g,o门参数，组成的数据集合
    #############################################################################
    def lstm_step_forward(self, x, prev_h, prev_c, Wx, Wh, b):
        """
        LSTM计算技巧说明 ：
        i_t = σ(W_xi*x_t + W_hi*h_(t-1) + b_i)
        f_t = σ(W_xf*x_t + W_hf*h_(t-1) + b_f)
        o_t = σ(W_xo*x_t + W_ho*h_(t-1) + b_o)
        c^_t = tanh(W_xc*x_t + W_hc*h_(t-1) + b_c)
        // g_t = tanh(W_ig*x_t + b_ig + W_hg*h_(t-1) + b_hg)
        // c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t
        #此处说明LSTM如何解决梯度消失原因，c_(t-1)表示过去信息，c^_t表示当前信息，此时c_t和c_(t-1)是线性关系而不再是乘积关系
        c_t = f_t ⊙ c_(t-1) + i_t ⊙ c^_t 
        h_t = o_t ⊙ tanh(c_t)
        通过前4个表达式可以看出，其实是x和h与f,i,g,o门对应的权重矩阵Wx和Wh进行了相同的矩阵运算，只是使用的权重矩阵不同，
        所以我们可以构建一个4倍大小的W，将f,i,g,o门对应的4个W矩阵拼接起来，计算之后再将4个矩阵分别分离出来
        这样可以减少计算量
        """
        next_h, next_c, cache = None, None, None
        # prev_h.shape(32, 30)
        H = prev_h.shape[1]
        # 合并之后的i,f,o,g在这里可以统一计算
        # 1.matmul作矩阵乘法(32, 120)=(32, 2)⊙(2, 120) + (32, 30)⊙(30, 120) + (120,) 
        # 2.matmul作矩阵乘法(32, 120)=(32, 30)⊙(30, 120) + (32, 30)⊙(30, 120) + (120,) 
        # 3.matmul作矩阵乘法(32, 120)=(32, 30)⊙(30, 120) + (32, 30)⊙(30, 120) + (120,) 
        z = Tools.matmul(x, Wx) + Tools.matmul(prev_h, Wh) + b

        # 之前将i,f,o,g四个矩阵合并了，这里将z(32,120)拆分4块进行计算，i,f,o,g的shape都是(32, 30)
        # 计算方式见注释“计算技巧部分” of shape(N,H)
        i = Tools.sigmoid(z[:,    :   H])
        f = Tools.sigmoid(z[:,  H : 2*H])
        o = Tools.sigmoid(z[:,2*H : 3*H])
        g = np.tanh(      z[:,3*H :    ])
        # next_c(32, 30) = (32, 30)*(32, 30) + (32, 30)*(32, 30)
        next_c = f * prev_c + i * g
        # next_h(32, 30)
        next_h = o * np.tanh(next_c)
        # i,f,o,g门产生的中间参数
        cache = (x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_c)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return next_h, next_c, cache
    
    #############################################################################
    # TODO: 每一个Timestep中的LSTM反向传播具体算法实现 
    # Inputs:
    # - dnext_h: 反向传播参数H_t(N, H)
    # - dnext_c: 反向传播参数C_t(N, H)
    # - cache: 前向传播过程中产生的中间参数
    # Returns a tuple of:
    # - dx: 反向传播到x的误差(N, D)
    # - dprev_h: 反向传播到H_{t-1}的误差(N, H)
    # - dprev_c: 反向传播到C_{t-1}的误差(N, H)
    # - dWx: 输入x->控制门的梯度矩阵(D, 4H)
    # - dWh: 状态h->控制门的梯度矩阵(H, 4H)
    # - db: 偏置项梯度(4H,)
    #############################################################################
    def lstm_step_backward(self, dnext_h, dnext_c, cache):
        """
        LSTM计算技巧说明 ：
        i_t = σ(W_xi*x_t + W_hi*h_(t-1) + b_i)
        f_t = σ(W_xf*x_t + W_hf*h_(t-1) + b_f)
        o_t = σ(W_xo*x_t + W_ho*h_(t-1) + b_o)
        c^_t = tanh(W_xc*x_t + W_hc*h_(t-1) + b_c)
        // g_t = tanh(W_ig*x_t + b_ig + W_hg*h_(t-1) + b_hg)
        // c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t
        c_t = f_t ⊙ c_(t-1) + i_t ⊙ c^_t
        h_t = o_t ⊙ tanh(c_t)
        1.四个参数矩阵di,df,do,dg 都要向前传播,所以合并成一个矩阵dz，在统一和权重矩阵做计算，
          所以我们可以构建一个4倍大小的W，将4个W矩阵拼接起来，计算之后再将4个矩阵分别分离出来，这样可以减少计算量
        2.针对ΔE/ΔC_t_new导数过程示例，除o以外，f i g的反向传播都包含两条求导链，来自C_t和h_t，所以先求出ΔE/ΔC_t_new导数，再代入f i g求导过程
          (1)ht = o_t ⊙ tanh(Ct)
          (2)ΔE/ΔC_t_new = ΔE/ΔC_t + ΔE/(Δh->ΔC)  #根据LSTM模型，可发现ΔE/ΔC_t有两条求导链，都要进行计算，其中一条就是ΔE/ΔC_t
          (3)ΔE/ΔC_t_new = ΔE/ΔC_t + ΔE/Δh*Δh/ΔC_t = ΔE/ΔC_t + ΔE/Δh * (o_t ⊙ tanh(Ct))' = ΔE/ΔC_t + ΔE/Δh * [o_t ⊙ (1-tanh2(C_t))]
        3.针对ΔE/ΔWo_t导数过程示例，先求ΔE/Δo_t，再求ΔE/ΔWo_t <=> [σ(x)]' = σ(x)*(1-σ(x))
          (1)ht = o_t ⊙ tanh(Ct)
          (2)ΔE/Δo_t = ΔE/Δh * Δh/Δo_t = ΔE/Δh * Δ[o_t ⊙ tanh(Ct)]/Δo_t = ΔE/Δh * tanh(Ct)
          (3)o_t = σ(W_xo*x_t + W_ho*h_(t-1) + b_o)
          (4)σ' = σ(1-σ)
          (5)ΔE/ΔWo_t = ΔE/Δh * Δh/Δo_t * Δo_t/ΔWo_t = ΔE/Δo_t * Δo_t/ΔWo_t = ΔE/Δo_t * (o_t)' = ΔE/Δo_t * [σ(W...)]' = (ΔE/Δo_t) * {σ(W+...) * [1-σ(W+...)]} = (ΔE/Δo_t) * o_t * (1-o_t)
          (6)ΔE/ΔWo_t = (ΔE/Δo_t) * o_t * (1-o_t) = ΔE/Δh * tanh(Ct) * o_t * (1-o_t)
        4.针对ΔE/ΔWc^_t(即ΔE/ΔWg_t)导数过程示例，先求ΔE/Δc^_t(即ΔE/Δg_t)，再求ΔE/ΔWc^_t = ΔE/ΔWg_t <=> [tanh(x)]' = 1-tanh2(x)
          (1)c_t = f_t ⊙ c_(t-1) + i_t ⊙ c^_t
          (2)ΔE/Δg_t = ΔE/Δc_t * Δc_t/Δg_t = ΔE/Δc_t * Δ[f_t ⊙ c_(t-1) + i_t ⊙ g_t]/Δg_t = ΔE/Δc_t * i_t
          (3)g_t = tanh(W_ig*x_t + b_ig + W_hg*h_(t-1) + b_hg)
          (4)tanh'(x) = 1-tanh2(x)
          (5)ΔE/ΔWg_t = ΔE/Δc_t * Δc_t/Δg_t * Δg_t/ΔWg_t = ΔE/Δg_t * Δg_t/ΔWg_t = ΔE/Δg_t * (g_t)' = ΔE/Δg_t * [tanh(W...)]' = (ΔE/Δc_t * i_t) * (1-tanh2(W...)) = (ΔE/Δc_t * i_t) * (1 - g_t^2)
        5.在求出上述三个导数ΔE/ΔC_t_new ~ dnext_c、ΔE/ΔWo_t ~ do、ΔE/ΔWc^_t ~ dg后，代入后续表达式即可
        """
        dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
        #x(32, 30)prev_h(32, 30)(32, 30)(30, 120)(30, 120)i(32, 30)(32, 30)(32, 30)(32, 30)(32, 30)
        x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_c = cache

        # ΔE/ΔC_t_new = ΔE/ΔC_t + (Δh->ΔC)，每个LSTM单元反向传播起始点包含C_t和H_t两条路径，所以此处进行了一个求和操作
        # next_h=o*np.tanh(next_c)，将求出的Δh/ΔC_t代入上式可得 => ΔE/(Δh->ΔC)=ΔE/Δh*Δh/ΔC_t = o*(1-np.tanh(next_c)**2)*ΔE/Δh
        dnext_c = dnext_c + o * (1 - np.tanh(next_c) ** 2) * dnext_h  
        # next_c = f*prev_c + i*g
        di = dnext_c * g
        df = dnext_c * prev_c
        # next_h = o*np.tanh(next_c)
        do = dnext_h * np.tanh(next_c)
        dg = dnext_c * i
        # next_c = f*prev_c + i*g
        dprev_c = f * dnext_c
        # 水平合并figo四部分(32, 120)
        dz = np.hstack((i * (1 - i) * di, f * (1 - f) * df, o * (1 - o) * do, (1 - g ** 2) * dg))  

        # 1.(32, 120)*(30,120).T = (32, 30) 
        # 2.(32, 120)*(30,120).T = (32, 30) 
        # 3.(32, 120)*(2,120).T=(32,2)
        dx = Tools.matmul(dz, Wx.T)
        # 1.(32, 120)*(30,120).T = (32, 30)
        # 2.(32, 120)*(30,120).T = (32, 30)
        # 3.(32, 120)*(30,120).T = (32, 30) 
        dprev_h = Tools.matmul(dz, Wh.T)
        # 1.(32, 30).T * (32, 120)=(30, 120)
        # 2.(32, 30).T * (32, 120)=(30, 120)
        # 3.(32,2).T * (32, 120)=(2,120)
        dWx = Tools.matmul(x.T, dz)
        # 1.(32, 30).T * (32, 120)=(30, 120)
        # 2.(32, 30).T * (32, 120)=(30, 120)
        # 3.(32, 30).T * (32, 120)=(30, 120)
        dWh = Tools.matmul(prev_h.T, dz)
        # 1,2,3:(32, 120)=>(120,)
        db = np.sum(dz, axis=0)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return dx, dprev_h, dprev_c, dWx, dWh, db


#最后一层concate,输出N*T*2H
class BiLSTMLayer(object):
    # N,H,L和优化器在初始化时定义
    # T作为X的一个维度传进来
    # tanh和sigmoid的前反向传播在类内部定义。
    # 直接输出分类维度
    def __init__(self, LName, miniBatchesSize, nodesNum, layersNum,
                 optimizerCls, optmParams, dropoutRRate, dataType, init_rng):
        # 初始化超参数
        self.name = LName
        self.miniBatchesSize = miniBatchesSize
        self.nodesNum = nodesNum
        self.layersNum = layersNum
        self.dataType = dataType
        self.init_rng = init_rng
        self.isInited = False  # 初始化标志

        # dropout 的保留率
        self.dropoutRRate = dropoutRRate
        self.dropoutMask = []

        self.out = []
        self.optimizerObjs = [optimizerCls(optmParams, dataType) for i in range(layersNum)]
        # 初始化w,u,b 和对应偏置,维度，层次和节点个数传参进去。但是没有T，所以不能创建参数
        # 返回的是一个组合结构,按层次（数组）划分的U、W，字典
        # 改为放在首batch X传入时lazy init
        self.lstmParams = []

        # 保存各层中间产出的 st和f(st)，用于前向和反向传播
        # 不需要，已经在前反向传播中保留
        self.deltaPrev = []  # 上一层激活后的误差输出

    def _initNnWeight(self, D, H, layersNum, dataType):

        # 层次
        lstmParams = []
        for layer in range(layersNum):
            Wh = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 4 * H)).astype(dataType)
            iWh = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 4 * H)).astype(dataType)
            if (0 == layer):
                Wx = np.random.uniform(-1 * self.init_rng, self.init_rng, (D, 4 * H)).astype(dataType)
                iWx = np.random.uniform(-1 * self.init_rng, self.init_rng, (D, 4 * H)).astype(dataType)
            else:
                Wx = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 4 * H)).astype(dataType)
                iWx = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 4 * H)).astype(dataType)
            b = np.zeros(4 * H, dataType)
            ib = np.zeros(4 * H, dataType)

            lstmParams.append({'Wx': Wx, 'Wh': Wh, 'b': b,
                               'iWx': iWx, 'iWh': iWh, 'ib': ib
                               # , 'U': U, 'V': V, 'bc': bc
                               })
        self.lstmParams = lstmParams

    # 预测时前向传播
    def fp(self, input):
        out_tmp = self.inference(input)
        self.out, self.dropoutMask = Tools.dropout4rnn(out_tmp, self.dropoutRRate)
        return self.out

    def inference(self, x):
        N, T, D = x.shape
        H = self.nodesNum
        L = self.layersNum
        # lazy init
        if (False == self.isInited):
            self._initNnWeight(D, H, L, self.dataType)
            self.isInited = True

        # 缓存已经存入rnnParams里了,此处只需要返回输出结果(N,T,H)
        h = self.lstm_forward(x)
        # N进 v 1出 模型，只保留时序最后的一项输出
        # self.out = h[:,-1,:]
        self.out = h
        return self.out

    # 反向传播方法(误差和权参)
    def bp(self, input, delta_ori, lrt):

        if self.dropoutRRate == 1:
            delta = delta_ori
        else:
            delta = delta_ori * self.dropoutMask

        # dw是一个数组，对应结构的多层，每层的dw,dh,db,dh0表示需要参数梯度
        # N, T, D = input.shape
        # H = delta.shape[1]
        # 只有最后一个T填delta，其余的dh梯度设置为0
        # dh = np.zeros((N, T, H), self.dataType)
        # dh[:,-1,:] = delta
        dh = delta
        dx, dweight = self.lstm_backward(dh)

        # 根据梯度更新参数
        self.bpWeights(dweight, lrt)

        return dx

    # 计算反向传播权重梯度w,b
    def bpWeights(self, dw, lrt):

        L = self.layersNum
        for l in range(L):
            w = (self.lstmParams[l]['Wx'], self.lstmParams[l]['Wh'], self.lstmParams[l]['b'],
                 self.lstmParams[l]['iWx'], self.lstmParams[l]['iWh'], self.lstmParams[l]['ib']
                 )
            self.optimizerObjs[l].getUpdWeights(w, dw[L - 1 - l], lrt)

    def lstm_forward(self, x):
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an input
        sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
        size of H, and we work over a minibatch containing N sequences. After running
        the LSTM forward, we return the hidden states for all timesteps.

        Note that the initial cell state is passed as input, but the initial cell
        state is set to zero. Also note that the cell state is not returned; it is
        an internal variable to the LSTM and is not accessed from outside.

        Inputs:
        - x: Input data of shape (N, T, D)
        - h0: Initial hidden state of shape (N, H)
        - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
        - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
        - b: Biases of shape (4H,)

        Returns a tuple of:
        - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
        - cache: Values needed for the backward pass.
        """
        #############################################################################
        # TODO: Implement the forward pass for an BiLSTM over an entire timeseries.   #
        # You should use the lstm_step_forward function that you just defined.      #
        # 首层，x(N,T,D), 向上变成xh(N,T,H)
        # 首层 Wx(D,H),   向上变成Wxh(H,H)
        #############################################################################
        N, T, D = x.shape
        L = self.layersNum
        H = int(self.lstmParams[0]['b'].shape[0] / 4)  # 取整
        xh = x  # 首层输入是x
        ixh = x # 反向
        for layer in range(L):
            h0 = np.zeros((N, H))
            c0 = np.zeros((N, H))

            # 右向
            h = np.zeros((N, T, H))
            c = np.zeros((N, T, H))
            cache = []

            # 左向
            ih = np.zeros((N, T, H))
            ic = np.zeros((N, T, H))
            icache = []
            for t in range(T):
                # 右向
                h[:, t, :], c[:, t, :], tmp_cache = self.lstm_step_forward(xh[:, t, :],
                                                                           h[:, t - 1, :] if t > 0 else h0,
                                                                           c[:, t - 1, :] if t > 0 else c0,
                                                                           self.lstmParams[layer]['Wx'],
                                                                           self.lstmParams[layer]['Wh'],
                                                                           self.lstmParams[layer]['b'])
                cache.append(tmp_cache)

                # 左向,
                # 若此处ih和x的下标保持一致，均由大到小排列，后续无需倒排,提高效率
                ih[:, T - 1 - t, :], ic[:, T - 1 - t, :], tmp_icache = self.lstm_step_forward(ixh[:, T - 1 - t, :],
                                                                              ih[:, T - t, :] if t > 0 else h0,
                                                                              ic[:, T - t, :] if t > 0 else c0,
                                                                              self.lstmParams[layer]['iWx'],
                                                                              self.lstmParams[layer]['iWh'],
                                                                              self.lstmParams[layer]['ib'])

                # icache下标和ih下标是反向对应的                                                              self.lstmParams[layer]['ib'])
                icache.append(tmp_icache)

            # 右向
            self.lstmParams[layer]['h'] = h
            self.lstmParams[layer]['c'] = c
            self.lstmParams[layer]['cache'] = cache

            # 左向
            self.lstmParams[layer]['ih'] = ih
            self.lstmParams[layer]['ic'] = ic
            self.lstmParams[layer]['icache'] = icache

            # Batch * TimeStep * H
            xh = h
            ixh = ih
            self.lstmParams[layer]['xh'] = xh
            self.lstmParams[layer]['ixh'] = ixh

        xh_final = np.concatenate((xh,ixh),axis=2) # 在H维度上做拼接
        self.lstmParams[layer]['xh_final'] = xh_final

        return xh_final

    def lstm_backward(self, dh_all):
        """
        Backward pass for an BiLSTM over an entire sequence of data.]

        Inputs:
        - dh_all: Upstream gradients of hidden states, of shape (N, T, 2*H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data of shape (N, T, D)
        - dh0: Gradient of initial hidden state of shape (N, H)
        - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        #############################################################################
        # TODO: Implement the backward pass for an BiLSTM over an entire timeseries.  #
        # You should use the lstm_step_backward function that you just defined.     #
        #############################################################################
        N, T, H_time_2 = dh_all.shape #得到的误差是batch *T* 2H
        H = int(H_time_2 / 2)

        x, _, _, _, _, _, _, _, _, _ = self.lstmParams[0]['cache'][0]

        D = x.shape[1] # 单个时间步上维度
        dh = dh_all[:,:,0:H]
        dih = dh_all[:,:,H:2*H]
        dweights = []

        for layer in range(self.layersNum - 1, -1, -1):

            dh_prevl = dh
            dih_prevl = dih

            DH = D if layer == 0 else H
            # 右向
            dx = np.zeros((N, T, DH))

            cache = self.lstmParams[layer]['cache']
            dWx = np.zeros((DH, 4 * H))
            dWh = np.zeros((H, 4 * H))
            db = np.zeros((4 * H))
            dprev_h = np.zeros((N, H))
            dprev_c = np.zeros((N, H))

            # 左向
            dix = np.zeros((N, T, DH))
            icache = self.lstmParams[layer]['icache']
            diWx = np.zeros((DH, 4 * H))
            diWh = np.zeros((H, 4 * H))
            dib = np.zeros((4 * H))
            dprev_ih = np.zeros((N, H))
            dprev_ic = np.zeros((N, H))

            for t in range(T - 1, -1, -1):
                # 右向
                dx[:, t, :], dprev_h, dprev_c, dWx_t, dWh_t, db_t = self.lstm_step_backward(dh_prevl[:, t, :] + dprev_h,
                                                                                            dprev_c,
                                                                                            cache[t])  # 注意此处的叠加
                dWx += dWx_t
                dWh += dWh_t
                db += db_t

                # fwd选择ih和输入x的下标一致对应，且之后合并前馈时，ih按时间步一致再前馈
                # bp时,按照时间步倒序回传,dih从小到大回传
                dix[:, T - 1 - t, :], dprev_ih, dprev_ic, diWx_t, diWh_t, db_it = self.lstm_step_backward(dih_prevl[:, T - 1 - t, :] + dprev_ih,
                                                                                                  dprev_ic,
                                                                                                  # icache[T - 1 - t])  # 注意此处的叠加
                                                                                                  icache[t])  # 注意此处的叠加
                diWx += diWx_t
                diWh += diWh_t
                dib += db_it

            dweight = (dWx, dWh, db, diWx, diWh, dib)
            dweights.append(dweight)

            # 本层得出的dx，作为下一层的误差输入
            dh = dx
            dih = dix
        # 第一层，正反两个方向的误差相加，得到总的dx返回上一层
        # 如果rnn是第一层，则误差不需要继续向上传递
        # 返回x误差和各层参数误差

        dh_t_all = dh + dih # 合并得到dx
        return dh_t_all, dweights

    def lstm_step_forward(self, x, prev_h, prev_c, Wx, Wh, b):
        """
        Forward pass for a single timestep of an LSTM.

        The input data has dimension D, the hidden state has dimension H, and we use
        a minibatch size of N.

        Note that a sigmoid() function has already been provided for you in this file.

        Inputs:
        - x: Input data, of shape (N, D)
        - prev_h: Previous hidden state, of shape (N, H)
        - prev_c: previous cell state, of shape (N, H)
        - Wx: Input-to-hidden weights, of shape (D, 4H)
        - Wh: Hidden-to-hidden weights, of shape (H, 4H)
        - b: Biases, of shape (4H,)

        Returns a tuple of:
        - next_h: Next hidden state, of shape (N, H)
        - next_c: Next cell state, of shape (N, H)
        - cache: Tuple of values needed for backward pass.
        """
        next_h, next_c, cache = None, None, None
        #############################################################################
        # TODO: Implement the forward pass for a single timestep of an LSTM.        #
        # You may want to use the numerically stable sigmoid implementation above.
        # 首层，x(N,T,D), 向上变成xh(N,T,H)
        # 首层 Wx(D,H),   向上变成Wxh(H,H)
        #############################################################################
        H = prev_h.shape[1]
        # z , of shape(N,4H)
        z = Tools.matmul(x, Wx) + Tools.matmul(prev_h, Wh) + b

        # of shape(N,H)
        i = Tools.sigmoid(z[:, :H])
        f = Tools.sigmoid(z[:, H:2 * H])
        o = Tools.sigmoid(z[:, 2 * H:3 * H])
        g = np.tanh(z[:, 3 * H:])
        next_c = f * prev_c + i * g
        next_h = o * np.tanh(next_c)

        cache = (x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_c)

        return next_h, next_c, cache

    def lstm_step_backward(self, dnext_h, dnext_c, cache):
        """
        Backward pass for a single timestep of an LSTM.

        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass for a single timestep of an LSTM.       #
        #                                                                           #
        # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
        # the output value from the nonlinearity.                                   #
        #############################################################################
        x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_c = cache

        dnext_c = dnext_c + o * (1 - np.tanh(next_c) ** 2) * dnext_h  # next_h = o*np.tanh(next_c)
        di = dnext_c * g  # next_c = f*prev_c + i*g
        df = dnext_c * prev_c  # next_c = f*prev_c + i*g
        do = dnext_h * np.tanh(next_c)  # next_h = o*np.tanh(next_c)
        dg = dnext_c * i  # next_h = o*np.tanh(next_c)
        dprev_c = f * dnext_c  # next_c = f*prev_c + i*g
        dz = np.hstack((i * (1 - i) * di, f * (1 - f) * df, o * (1 - o) * do, (1 - g ** 2) * dg))  # 共四部分

        dx = Tools.matmul(dz, Wx.T)
        dprev_h = Tools.matmul(dz, Wh.T)
        dWx = Tools.matmul(x.T, dz)
        dWh = Tools.matmul(prev_h.T, dz)

        db = np.sum(dz, axis=0)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return dx, dprev_h, dprev_c, dWx, dWh, db

# GRU 类
class GRULayer(object):

    def __init__(self, LName, miniBatchesSize, nodesNum, layersNum,
                 optimizerCls, optmParams, dropoutRRate, dataType, init_rng):
        # 初始化超参数
        self.name = LName
        self.miniBatchesSize = miniBatchesSize
        self.nodesNum = nodesNum
        self.layersNum = layersNum
        # self.optimizer = optimizer
        self.dataType = dataType
        self.init_rng = init_rng
        self.isInited = False  # 初始化标志

        # dropout 的保留率
        self.dropoutRRate = dropoutRRate
        self.dropoutMask = []

        self.out = []
        self.optimizerObjs = [optimizerCls(optmParams, dataType) for i in range(layersNum)]
        # 初始化w,u,b 和对应偏置,维度，层次和节点个数传参进去。但是没有T，所以不能创建参数
        # 返回的是一个组合结构,按层次（数组）划分的U、W，字典
        # 改为放在首batch X传入时lazy init
        self.gruParams = []

        # 保存各层中间产出的 st和f(st)，用于前向和反向传播
        # 不需要，已经在前反向传播中保留
        self.deltaPrev = []  # 上一层激活后的误差输出

    # N,H,L和优化器在初始化时定义
    # T作为X的一个维度传进来
    # tanh和sigmoid的前反向传播在类内部定义。
    def _initNnWeight(self, D, H, layersNum, dataType):

        # 层次
        gruParams = []
        for layer in range(layersNum):
            Wzh = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 2 * H)).astype(dataType)
            War = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, H)).astype(dataType)
            if (0 == layer):
                Wzx = np.random.uniform(-1 * self.init_rng, self.init_rng, (D, 2 * H)).astype(dataType)
                Wax = np.random.uniform(-1 * self.init_rng, self.init_rng, (D, H)).astype(dataType)
            else:
                Wzx = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 2 * H)).astype(dataType)
                Wax = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, H)).astype(dataType)
            bz = np.zeros(2 * H, dataType)
            ba = np.zeros(H, dataType)
            gruParams.append({'Wzx': Wzx, 'Wzh': Wzh, 'bz': bz, 'Wax': Wax, 'War': War, 'ba': ba})

        self.gruParams = gruParams

    def fp(self, input):
        out_tmp = self.inference(input)
        self.out, self.dropoutMask = Tools.dropout4rnn(out_tmp, self.dropoutRRate)

        return self.out

    # 预测时前向传播
    def inference(self, x):
        N, T, D = x.shape
        H = self.nodesNum
        L = self.layersNum
        # lazy init
        if (False == self.isInited):
            self._initNnWeight(D, H, L, self.dataType)
            self.isInited = True

        # 缓存已经存入rnnParams里了,此处只需要返回输出结果(N,T,H)
        h = self.gru_forward(x)
        # N进 v 1出 模型，只保留时序最后的一项输出
        # self.out = h[:,-1,:]
        self.out = h
        return self.out

    # 反向传播方法(误差和权参)
    # TODO 实现反向传播逻辑，先按照时间，再按照层次，再更新Wx/Wf/b/V/bv 及偏置的反向传播梯度
    def bp(self, input, delta_ori, lrt):

        if self.dropoutRRate == 1:
            delta = delta_ori
        else:
            delta = delta_ori * self.dropoutMask

        # dw是一个数组，对应结构的多层，每层的dw,dh,db,dh0表示需要参数梯度
        N, T, D = input.shape
        H = delta.shape[1]
        # 只有最后一个T填delta，其余的dh梯度设置为0
        dh = np.zeros((N, T, H), self.dataType)
        # dh[:,-1,:] = delta
        dh = delta
        dx, dweight = self.gru_backward(dh)

        # 根据梯度更新参数
        self.bpWeights(dweight, lrt)

        return dx

    # 计算反向传播权重梯度w,b
    def bpWeights(self, dw, lrt):

        L = self.layersNum
        for l in range(L):
            w = (self.gruParams[l]['Wzx'], self.gruParams[l]['Wzh'], self.gruParams[l]['bz'], self.gruParams[l]['Wax'],
                 self.gruParams[l]['War'], self.gruParams[l]['ba'])
            # self.gruParams[l]['Wzx'], self.gruParams[l]['Wzh'], self.gruParams[l]['bz'],self.gruParams[l]['Wax'], self.gruParams[l]['War'], self.gruParams[l]['ba'] = self.optimizerObjs[l].getUpdWeights(w, dw[L-1-l], lrt)
            self.optimizerObjs[l].getUpdWeights(w, dw[L - 1 - l], lrt)

    def gru_forward(self, x):
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an input
        sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
        size of H, and we work over a minibatch containing N sequences. After running
        the LSTM forward, we return the hidden states for all timesteps.

        Note that the initial cell state is passed as input, but the initial cell
        state is set to zero. Also note that the cell state is not returned; it is
        an internal variable to the LSTM and is not accessed from outside.

        Inputs:
        - x: Input data of shape (N, T, D)
        - h0: Initial hidden state of shape (N, H)
        - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
        - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
        - b: Biases of shape (4H,)

        Returns a tuple of:
        - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
        - cache: Values needed for the backward pass.
        """
        h, cache = None, None
        #############################################################################
        # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
        # You should use the lstm_step_forward function that you just defined.      #
        # 首层，x(N,T,D), 向上变成xh(N,T,H)
        # 首层 Wx(D,H),   向上变成Wxh(H,H)
        #############################################################################
        N, T, D = x.shape
        L = self.layersNum
        H = self.gruParams[0]['ba'].shape[0]  # 取整
        xh = x  # 首层输入是x
        for layer in range(L):
            h = np.zeros((N, T, H))
            h0 = np.zeros((N, H))
            cache = []
            for t in range(T):
                h[:, t, :], tmp_cache = self.gru_step_forward(xh[:, t, :], h[:, t - 1, :] if t > 0 else h0,
                                                              self.gruParams[layer]['Wzx'],
                                                              self.gruParams[layer]['Wzh'],
                                                              self.gruParams[layer]['bz'],
                                                              self.gruParams[layer]['Wax'],
                                                              self.gruParams[layer]['War'],
                                                              self.gruParams[layer]['ba'],
                                                              )
                cache.append(tmp_cache)
            xh = h  # 之后以h作为xh作为跨层输入
            ##############################################################################
            #                               END OF YOUR CODE                             #
            ##############################################################################
            self.gruParams[layer]['h'] = h
            self.gruParams[layer]['cache'] = cache

        return h

    def gru_backward(self, dh):
        """
        Backward pass for an LSTM over an entire sequence of data.]

        Inputs:
        - dh: Upstream gradients of hidden states, of shape (N, T, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data of shape (N, T, D)
        - dh0: Gradient of initial hidden state of shape (N, H)
        - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dh0, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
        # You should use the lstm_step_backward function that you just defined.     #
        #############################################################################
        N, T, H = dh.shape
        x, _, _, _, _, _, _, _, _, _ = self.gruParams[0]['cache'][0]
        D = x.shape[1]

        dh_prevl = dh
        # 保存各层dwh,dwx,和db
        dweights = []

        for layer in range(self.layersNum - 1, -1, -1):
            # 得到前向传播保存的cache数组
            cache = self.gruParams[layer]['cache']

            DH = D if layer == 0 else H
            dx = np.zeros((N, T, DH))
            dWzx = np.zeros((DH, 2 * H))
            dWzh = np.zeros((H, 2 * H))
            dbz = np.zeros((2 * H))

            dWax = np.zeros((DH, H))
            dWar = np.zeros((H, H))
            dba = np.zeros((H))

            dprev_h = np.zeros((N, H))

            for t in range(T - 1, -1, -1):
                dx[:, t, :], dprev_h, dWzx_t, dWzh_t, dbz_t, dWax_t, dWar_t, dba_t = self.gru_step_backward(
                    dh_prevl[:, t, :] + dprev_h,
                    cache[t])  # 注意此处的叠加
                dWzx += dWzx_t
                dWzh += dWzh_t
                dbz += dbz_t

                dWax += dWax_t
                dWar += dWar_t
                dba += dba_t
            # 本层得出的dx，作为下一层的prev_l
            dh_prevl = dx

            dweight = (dWzx, dWzh, dbz, dWax, dWar, dba)
            dweights.append(dweight)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        # 返回x误差和各层参数误差
        return dx, dweights

    def gru_step_forward(self, x, prev_h, Wzx, Wzh, bz, Wax, War, ba):
        """
        Forward pass for a single timestep of an LSTM.

        The input data has dimension D, the hidden state has dimension H, and we use
        a minibatch size of N.

        Note that a sigmoid() function has already been provided for you in this file.

        Inputs:
        - x: Input data, of shape (N, D)
        - prev_h: Previous hidden state, of shape (N, H)
        - prev_c: previous cell state, of shape (N, H)
        - Wzx: Input-to-hidden weights, of shape (D, 4H)
        - Wh: Hidden-to-hidden weights, of shape (H, 4H)
        - b: Biases, of shape (4H,)

        Returns a tuple of:
        - next_h: Next hidden state, of shape (N, H)
        - next_c: Next cell state, of shape (N, H)
        - cache: Tuple of values needed for backward pass.
        """
        next_h, cache = None, None
        #############################################################################
        # TODO: Implement the forward pass for a single timestep of an LSTM.        #
        # You may want to use the numerically stable sigmoid implementation above.
        # 首层，x(N,T,D), 向上变成xh(N,T,H)
        # 首层 Wx(D,H),   向上变成Wxh(H,H)
        #############################################################################
        H = prev_h.shape[1]
        # z_hat, of shape(N,4H)
        z_hat = Tools.matmul(x, Wzx) + Tools.matmul(prev_h, Wzh) + bz

        # of shape(N,H)
        r = Tools.sigmoid(z_hat[:, :H])
        z = Tools.sigmoid(z_hat[:, H:2 * H])

        a = Tools.matmul(x, Wax) + Tools.matmul(r * prev_h, War) + ba

        next_h = prev_h * (1. - z) + z * np.tanh(a)

        cache = (x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return next_h, cache

    def gru_step_backward(self, dnext_h, cache):
        """
        Backward pass for a single timestep of an LSTM.

        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass for a single timestep of an LSTM.       #
        #                                                                           #
        # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
        # the output value from the nonlinearity.                                   #
        #############################################################################
        x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a = cache

        N, D = x.shape
        H = dnext_h.shape[1]

        z_hat_H1 = z_hat[:, :H]
        z_hat_H2 = z_hat[:, H:2 * H]
        # delta
        tanha = np.tanh(a)
        dh = dnext_h
        da = dh * z * (1. - tanha ** 2)
        dh_prev_1 = dh * (1. - z)
        # dz = dh*(tanha-prev_h)
        dz_hat_2 = dh * (tanha - prev_h) * (z * (1. - z))

        dhat_a = Tools.matmul(da, War.T)
        dr = dhat_a * prev_h

        dx_1 = Tools.matmul(da, Wax.T)
        dh_prev_2 = dhat_a * r  # da* Tools.matmul(r,War.T)
        dz_hat_1 = dr * (r * (1. - r))

        dz_hat = np.hstack((dz_hat_1, dz_hat_2))

        dx_2 = Tools.matmul(dz_hat_1, Wzx[:, :H].T)
        dh_prev_3 = Tools.matmul(dz_hat_1, Wzh[:, :H].T)

        dx_3 = Tools.matmul(dz_hat_2, Wzx[:, H:2 * H].T)
        dh_prev_4 = Tools.matmul(dz_hat_2, Wzh[:, H:2 * H].T)

        dprev_h = dh_prev_1 + dh_prev_2 + dh_prev_3 + dh_prev_4
        dx = dx_1 + dx_2 + dx_3

        dWax = Tools.matmul(x.T, da)
        dWar = Tools.matmul((r * prev_h).T, da)
        dba = np.sum(da, axis=0)

        dWzx = Tools.matmul(x.T, dz_hat)
        dWzh = Tools.matmul(prev_h.T, dz_hat)

        dbz = np.sum(dz_hat, axis=0)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba

    def gru_step_backward_succ(self, dnext_h, cache):
        """
        Backward pass for a single timestep of an LSTM.

        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass for a single timestep of an LSTM.       #
        #                                                                           #
        # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
        # the output value from the nonlinearity.                                   #
        #############################################################################
        x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a = cache

        N, D = x.shape
        H = dnext_h.shape[1]

        z_hat_H1 = z_hat[:, :H]
        z_hat_H2 = z_hat[:, H:2 * H]
        # delta
        tanha = np.tanh(a)
        dh = dnext_h
        da = dh * z * (1. - tanha ** 2)
        dh_prev_1 = dh * (1. - z)
        # dz = dh * (z+tanha)
        # dz = dh*tanha+1.-dh*(1.-z)*prev_h
        # dz = dh*tanha+1.-dh*prev_h
        dz = dh * (tanha - prev_h)
        # dz_hat_2 = dz*(z*(1.-z))
        dz_hat_2 = dh * (tanha - prev_h) * (z * (1. - z))
        # dz_hat_2 = dz*(z_hat_H2*(1.-z_hat_H2))

        dhat_a = Tools.matmul(da, War.T)
        # dz_hat_2 = dhat_r * r
        dr = dhat_a * prev_h

        dx_1 = Tools.matmul(da, Wax.T)
        dh_prev_2 = dhat_a * r  # da* Tools.matmul(r,War.T)
        # dz_hat_1 = dh_prev_2 * (r * (1. - r))
        dz_hat_1 = dr * (r * (1. - r))
        # dz_hat_1 = prev_h * Tools.matmul(dh*z*(1-tanha**2), War.T)*(r*(1.-r))

        dz_hat = np.hstack((dz_hat_1, dz_hat_2))

        # dh_prev_3 = Tools.matmul(dz_hat_2,Wzh.T)
        # dx_2 = Tools.matmul(dz_hat_2,Wzx.T)
        dx_2 = Tools.matmul(dz_hat_1, Wzx[:, :H].T)
        # dh_prev_3 = Tools.matmul(dz_hat,Wzh.T)
        # dh_prev_3 = Tools.matmul(dz_hat_2,Wzh.T)
        dh_prev_3 = Tools.matmul(dz_hat_1, Wzh[:, :H].T)
        dx_23 = Tools.matmul(dz_hat, Wzx.T)

        # dx_3 = Tools.matmul(dz_hat_1,Wzx.T)
        dx_3 = Tools.matmul(dz_hat_2, Wzx[:, H:2 * H].T)
        # dh_prev_4 =Tools.matmul(dz_hat_1, Wzh.T)
        dh_prev_4 = Tools.matmul(dz_hat_2, Wzh[:, H:2 * H].T)
        # dx_3 = Tools.matmul(dz_hat,Wzx.T)
        # dh_prev_4 =Tools.matmul(dz_hat, Wzh.T)

        # dh_prev_34 = np.hstack((dh_prev_3, dh_prev_4))
        # dh_prev_34 = Tools.matmul(dh_prev_34,Wzh.T)
        dh_prev_34 = Tools.matmul(dz_hat, Wzh.T)
        # dprev_h = dh_prev_1+dh_prev_2+dh_prev_34 * 2. #dh_prev_3 + dh_prev_4
        # dx = dx_1 + dx_2*2. # +dx_3
        # dprev_h = dh_prev_1+dh_prev_2+dh_prev_34  #dh_prev_3 + dh_prev_4
        dprev_h = dh_prev_1 + dh_prev_2 + dh_prev_3 + dh_prev_4
        # dx = dx_1 + dx_23 # +dx_3
        dx = dx_1 + dx_2 + dx_3

        dWax = Tools.matmul(x.T, da)
        dWar = Tools.matmul((r * prev_h).T, da)
        dba = np.sum(da, axis=0)

        dWzx = Tools.matmul(x.T, dz_hat)
        dWzh = Tools.matmul(prev_h.T, dz_hat)
        dbz = np.sum(dz_hat, axis=0)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba

    def gru_step_backward_v2(self, dnext_h, cache):
        """
        Backward pass for a single timestep of an LSTM.

        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass for a single timestep of an LSTM.       #
        #                                                                           #
        # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
        # the output value from the nonlinearity.                                   #
        #############################################################################
        x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a = cache

        N, D = x.shape
        H = dnext_h.shape[1]

        z_hat_H1 = z_hat[:, :H]
        z_hat_H2 = z_hat[:, H:2 * H]
        # delta
        tanha = np.tanh(a)
        dh = dnext_h
        da = dh * z * (1. - tanha ** 2)
        dh_prev_1 = dh * (1. - z)
        # dz = dh * (z+tanha)
        # dz = dh*tanha+1.-dh*(1.-z)*prev_h
        # dz = dh*tanha+1.-dh*prev_h
        dz = dh * (tanha - prev_h)
        # dz_hat_2 = dz*(z*(1.-z))
        dz_hat_2 = dh * (tanha - prev_h) * (z * (1. - z))
        # dz_hat_2 = dz*(z_hat_H2*(1.-z_hat_H2))

        dhat_a = Tools.matmul(da, War.T)
        # dz_hat_2 = dhat_r * r
        dr = dhat_a * prev_h

        dx_1 = Tools.matmul(da, Wax.T)
        dh_prev_2 = dhat_a * r  # da* Tools.matmul(r,War.T)
        # dz_hat_1 = dh_prev_2 * (r * (1. - r))
        dz_hat_1 = dr * (r * (1. - r))
        # dz_hat_1 = prev_h * Tools.matmul(dh*z*(1-tanha**2), War.T)*(r*(1.-r))

        dz_hat = np.hstack((dz_hat_1, dz_hat_2))

        # dh_prev_3 = Tools.matmul(dz_hat_2,Wzh.T)
        # dx_2 = Tools.matmul(dz_hat_2,Wzx.T)
        # dh_prev_3 = Tools.matmul(dz_hat,Wzh.T)
        # dh_prev_3 = Tools.matmul(dz_hat_2,Wzh.T)
        dx_23 = Tools.matmul(dz_hat, Wzx.T)

        # dx_3 = Tools.matmul(dz_hat_1,Wzx.T)
        # dh_prev_4 =Tools.matmul(dz_hat_1, Wzh.T)
        # dx_3 = Tools.matmul(dz_hat,Wzx.T)
        # dh_prev_4 =Tools.matmul(dz_hat, Wzh.T)

        # dh_prev_34 = np.hstack((dh_prev_3, dh_prev_4))
        # dh_prev_34 = Tools.matmul(dh_prev_34,Wzh.T)
        dh_prev_34 = Tools.matmul(dz_hat, Wzh.T)
        # dprev_h = dh_prev_1+dh_prev_2+dh_prev_34 * 2. #dh_prev_3 + dh_prev_4
        # dx = dx_1 + dx_2*2. # +dx_3
        dprev_h = dh_prev_1 + dh_prev_2 + dh_prev_34  # dh_prev_3 + dh_prev_4
        dx = dx_1 + dx_23  # +dx_3

        dWax = Tools.matmul(x.T, da)
        dWar = Tools.matmul((r * prev_h).T, da)
        dba = np.sum(da, axis=0)

        dWzx = Tools.matmul(x.T, dz_hat)
        dWzh = Tools.matmul(prev_h.T, dz_hat)
        dbz = np.sum(dz_hat, axis=0)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba

    def gru_step_backward_v1(self, dnext_h, cache):
        """
        Backward pass for a single timestep of an LSTM.

        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass for a single timestep of an LSTM.       #
        #                                                                           #
        # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
        # the output value from the nonlinearity.                                   #
        #############################################################################
        x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a = cache

        N, D = x.shape
        H = dnext_h.shape[1]

        z_hat_H1 = z_hat[:, :H]
        z_hat_H2 = z_hat[:, H:2 * H]
        # delta
        tanha = np.tanh(a)
        dh = dnext_h
        da = dh * z * (1. - tanha * tanha)
        dh_prev_1 = dh * (1. - z)
        # dz = dh * (z+tanha)
        # dz = dh*tanha+1.-dh*(1.-z)*prev_h
        # dz = dh*tanha+1.-dh*prev_h
        dz = dh * (tanha - prev_h)
        dz_hat_2 = dz * (z * (1. - z))
        # dz_hat_2 = dz*(z_hat_H2*(1.-z_hat_H2))

        dhat_a = Tools.matmul(da, War.T)
        # dz_hat_2 = dhat_r * r
        dr = dhat_a * prev_h

        dx_1 = Tools.matmul(da, Wax.T)
        dh_prev_2 = dhat_a * r  # da* Tools.matmul(r,War.T)
        # dz_hat_1 = dh_prev_2 * (r * (1. - r))
        dz_hat_1 = dr * (r * (1. - r))

        dz_hat = np.hstack((dz_hat_1, dz_hat_2))

        # dh_prev_3 = Tools.matmul(dz_hat_2,Wzh.T)
        # dx_2 = Tools.matmul(dz_hat_2,Wzx.T)
        # dh_prev_3 = Tools.matmul(dz_hat,Wzh.T)
        # dh_prev_3 = Tools.matmul(dz_hat_2,Wzh.T)
        dx_2 = Tools.matmul(dz_hat, Wzx.T)

        # dx_3 = Tools.matmul(dz_hat_1,Wzx.T)
        # dh_prev_4 =Tools.matmul(dz_hat_1, Wzh.T)
        # dx_3 = Tools.matmul(dz_hat,Wzx.T)
        # dh_prev_4 =Tools.matmul(dz_hat, Wzh.T)

        # dh_prev_34 = np.hstack((dh_prev_3, dh_prev_4))
        # dh_prev_34 = Tools.matmul(dh_prev_34,Wzh.T)
        dh_prev_34 = Tools.matmul(dz_hat, Wzh.T)
        # dprev_h = dh_prev_1+dh_prev_2+dh_prev_34 * 2. #dh_prev_3 + dh_prev_4
        # dx = dx_1 + dx_2*2. # +dx_3
        dprev_h = dh_prev_1 + dh_prev_2 + dh_prev_34  # dh_prev_3 + dh_prev_4
        dx = dx_1 + dx_2  # +dx_3

        dWax = Tools.matmul(x.T, da)
        dWar = Tools.matmul((r * prev_h).T, da)
        dba = np.sum(da, axis=0)

        dWzx = Tools.matmul(x.T, dz_hat)
        dWzh = Tools.matmul(prev_h.T, dz_hat)
        dbz = np.sum(dz_hat, axis=0)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba

    def gru_step_backward_v0(self, dnext_h, cache):
        """
        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a = cache

        N, D = x.shape
        H = dnext_h.shape[1]

        z_hat_H1 = z_hat[:, :H]
        z_hat_H2 = z_hat[:, H:2 * H]
        # delta
        tanha = np.tanh(a)
        dh = dnext_h
        da = dh * z * (1. - tanha * tanha)
        dh_prev_1 = dh * (1. - z)
        dz = dh * (z + tanha)
        dz_hat_2 = dz * (z * (1. - z))

        d13 = np.matmul(da, War.T)
        dr = d13 * prev_h
        dx_1 = np.matmul(da, Wax.T)
        dh_prev_2 = d13 * r
        dz_hat_1 = dh_prev_2 * (r * (1. - r))

        dz_hat = np.hstack((dz_hat_1, dz_hat_2))

        dh_prev_3 = np.matmul(dz_hat, Wzh.T)
        dx_2 = np.matmul(dz_hat, Wzx.T)
        dx_3 = np.matmul(dz_hat, Wzx.T)
        dh_prev_4 = np.matmul(dz_hat, Wzh.T)
        dprev_h = dh_prev_1 + dh_prev_2 + dh_prev_3 + dh_prev_4
        dx = dx_1 + dx_2 + dx_3

        dWax = np.matmul(x.T, da)
        dWar = np.matmul((r * prev_h).T, da)
        dba = np.sum(da, axis=0)

        dWzx = np.matmul(x.T, dz_hat)
        dWzh = np.matmul(prev_h.T, dz_hat)
        dbz = np.sum(dz_hat, axis=0)

        return dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba

# concat 方式
class BiGRULayer(object):

    def __init__(self, LName, miniBatchesSize, nodesNum, layersNum,
                 optimizerCls, optmParams, dropoutRRate, dataType, init_rng):
        # 初始化超参数
        self.name = LName
        self.miniBatchesSize = miniBatchesSize
        self.nodesNum = nodesNum
        self.layersNum = layersNum
        self.dataType = dataType
        self.init_rng = init_rng
        self.isInited = False  # 初始化标志

        # dropout 的保留率
        self.dropoutRRate = dropoutRRate
        self.dropoutMask = []

        self.out = []
        self.optimizerObjs = [optimizerCls(optmParams, dataType) for i in range(layersNum)]
        # 初始化w,u,b 和对应偏置,维度，层次和节点个数传参进去。但是没有T，所以不能创建参数
        # 返回的是一个组合结构,按层次（数组）划分的U、W，字典
        # 改为放在首batch X传入时lazy init
        self.gruParams = []

        # 不需要保存各层中间产出的 st和f(st)，已经在前反向传播中保留，用于前向和反向传播，
        self.deltaPrev = []  # 上一层激活后的误差输出

    # N,H,L和优化器在初始化时定义
    # T作为X的一个维度传进来
    # tanh和sigmoid的前反向传播在类内部定义。
    def _initNnWeight(self, D, H, layersNum, dataType):

        # 层次
        gruParams = []
        for layer in range(layersNum):
            # 右向
            Wzh = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 2 * H)).astype(dataType)
            War = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, H)).astype(dataType)
            # 左向
            iWzh = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 2 * H)).astype(dataType)
            iWar = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, H)).astype(dataType)

            DH = D if layer == 0 else H
            # 右向
            Wzx = np.random.uniform(-1 * self.init_rng, self.init_rng, (DH, 2 * H)).astype(dataType)
            Wax = np.random.uniform(-1 * self.init_rng, self.init_rng, (DH, H)).astype(dataType)
            bz = np.zeros(2 * H, dataType)
            ba = np.zeros(H, dataType)

            # 左向
            iWzx = np.random.uniform(-1 * self.init_rng, self.init_rng, (DH, 2 * H)).astype(dataType)
            iWax = np.random.uniform(-1 * self.init_rng, self.init_rng, (DH, H)).astype(dataType)
            ibz = np.zeros(2 * H, dataType)
            iba = np.zeros(H, dataType)

            gruParams.append({'Wzx': Wzx, 'Wzh': Wzh, 'bz': bz, 'Wax': Wax, 'War': War, 'ba': ba,
                              'iWzx': iWzx, 'iWzh': iWzh, 'ibz': ibz, 'iWax': iWax, 'iWar': iWar, 'iba': iba
                              })
        self.gruParams = gruParams

    def fp(self, input):
        out_tmp = self.inference(input)
        self.out, self.dropoutMask = Tools.dropout4rnn(out_tmp, self.dropoutRRate)

        return self.out

    # 预测时前向传播
    def inference(self, x):
        N, T, D = x.shape
        H = self.nodesNum
        L = self.layersNum
        # lazy init
        if (False == self.isInited):
            self._initNnWeight(D, H, L, self.dataType)
            self.isInited = True

        # 缓存已经存入rnnParams里了,此处只需要返回输出结果(N,T,H)

        # N进 v 1出 模型，只保留时序最后的一项输出
        # h = self.gru_forward(x)
        # self.out = h[:,-1,:]

        # N进N出模型，全部输出
        self.out = self.gru_forward(x)
        return self.out

    # 反向传播方法(误差和权参)
    def bp(self, input, delta_ori, lrt):

        if self.dropoutRRate == 1:
            delta = delta_ori
        else:
            delta = delta_ori * self.dropoutMask

        # dw是一个数组，对应结构的多层，每层的dw,dh,db,dh0表示需要参数梯度
        # N v 1 只有最后一个T填delta，其余的dh梯度设置为0
        # N, T, D = input.shape
        # H = delta.shape[1]
        # dh = np.zeros((N, T, H), self.dataType)
        # dh[:,-1,:] = delta

        # N v N模型
        dh = delta
        dx, dweight = self.gru_backward(dh)

        # 根据梯度更新参数
        self.bpWeights(dweight, lrt)

        return dx

    # 计算反向传播权重梯度w,b
    def bpWeights(self, dw, lrt):

        L = self.layersNum
        for l in range(L):
            # w = (self.gruParams[l]['Wzx'], self.gruParams[l]['Wzh'], self.gruParams[l]['bz'],
            #      self.gruParams[l]['Wax'], self.gruParams[l]['War'], self.gruParams[l]['ba'],
            #      self.gruParams[l]['iWzx'], self.gruParams[l]['iWzh'], self.gruParams[l]['ibz'],
            #      self.gruParams[l]['iWax'], self.gruParams[l]['iWar'], self.gruParams[l]['iba']
            #      # ,self.gruParams[l]['U'], self.gruParams[l]['V'], self.gruParams[l]['bc']
            #      )
            params=self.gruParams[l]
            w = (params['Wzx'], params['Wzh'], params['bz'],
                 params['Wax'], params['War'], params['ba'],
                 params['iWzx'], params['iWzh'], params['ibz'],
                 params['iWax'], params['iWar'], params['iba']
                 )

            # is_same0 = op.eq(w,w0)

            # 梯度倒序append到dw中
            self.optimizerObjs[l].getUpdWeights(w, dw[L - 1 - l], lrt)

    def bpWeights_v1(self, dw, lrt):

        L = self.layersNum
        for l in range(L):
            w = (self.gruParams[l]['Wzx'], self.gruParams[l]['Wzh'], self.gruParams[l]['bz'],
                 self.gruParams[l]['Wax'], self.gruParams[l]['War'], self.gruParams[l]['ba'],
                 self.gruParams[l]['iWzx'], self.gruParams[l]['iWzh'], self.gruParams[l]['ibz'],
                 self.gruParams[l]['iWax'], self.gruParams[l]['iWar'], self.gruParams[l]['iba']
                 # ,self.gruParams[l]['U'], self.gruParams[l]['V'], self.gruParams[l]['bc']
                 )
            w1 = (self.gruParams[l][element] for element in self.gruParams[l])
            w2 = (v for (k,v) in self.gruParams[l].items())
            is_same1 = op.eq(w,w1)
            is_same2 = op.eq(w, w2)

            # 梯度倒序append到dw中
            self.optimizerObjs[l].getUpdWeights(w, dw[L - 1 - l], lrt)

    def gru_forward(self, x):
        """
        Forward pass for an BiGRU over an entire sequence of data. We assume an input
        sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
        size of H, and we work over a minibatch containing N sequences. After running
        the LSTM forward, we return the hidden states for all timesteps.

        Note that the initial cell state is passed as input, but the initial cell
        state is set to zero. Also note that the cell state is not returned; it is
        an internal variable to the LSTM and is not accessed from outside.

        Inputs:
        - x: Input data of shape (N, T, D)
        - h0: Initial hidden state of shape (N, H)
        - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
        - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
        - b: Biases of shape (4H,)

        Returns a tuple of:
        - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
        - cache: Values needed for the backward pass.
        """

        #############################################################################
        # 首层，x(N,T,D), 向上变成xh(N,T,H)
        # 首层 Wx(D,H),   向上变成Wxh(H,H)
        #############################################################################
        N, T, D = x.shape
        L = self.layersNum
        H = self.gruParams[0]['ba'].shape[0]  # 取整
        xh = x  # 首层输入是x
        ixh= x  # 反向
        for layer in range(L):
            h0 = np.zeros((N, H))

            # 右向
            h = np.zeros((N, T, H))
            cache = []

            # 左向
            ih = np.zeros((N, T, H))
            icache = []
            for t in range(T):
                # 右向
                h[:, t, :], tmp_cache = self.gru_step_forward(xh[:, t, :],
                                                              h[:, t - 1, :] if t > 0 else h0,
                                                              self.gruParams[layer]['Wzx'],
                                                              self.gruParams[layer]['Wzh'],
                                                              self.gruParams[layer]['bz'],
                                                              self.gruParams[layer]['Wax'],
                                                              self.gruParams[layer]['War'],
                                                              self.gruParams[layer]['ba']
                                                              )
                cache.append(tmp_cache)

                # 左向
                ih[:, T - 1 - t , :], tmp_icache = self.gru_step_forward(ixh[:, T - 1 - t, :],
                                                              ih[:, T - t, :] if t > 0 else h0,
                                                              self.gruParams[layer]['iWzx'],
                                                              self.gruParams[layer]['iWzh'],
                                                              self.gruParams[layer]['ibz'],
                                                              self.gruParams[layer]['iWax'],
                                                              self.gruParams[layer]['iWar'],
                                                              self.gruParams[layer]['iba']
                                                              )
                # icache是下标和ih下标是返向的                                              )
                icache.append(tmp_icache)
            # 右向
            self.gruParams[layer]['h'] = h
            self.gruParams[layer]['cache'] = cache

            # 左向
            self.gruParams[layer]['ih'] = ih
            self.gruParams[layer]['icache'] = icache

            xh = h
            ixh = ih
        xh_final = np.concatenate((xh, ixh), axis=2)  # 在H维度上做拼接
        # self.gruParams[layer]['xh_final'] = xh_final
        return xh_final

    def gru_backward(self, dh_all):
        """
        Backward pass for an BiLSTM over an entire sequence of data.]

        Inputs:
        - dh: Upstream gradients of hidden states, of shape (N, T, 2H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data of shape (N, T, D)
        - dh0: Gradient of initial hidden state of shape (N, H)
        - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dh0, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        diWzx, diWzh, dibz, diWax, diWar, diba = None, None, None, None, None, None
        dU, dV, dbc = None, None, None
        #############################################################################
        # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
        # You should use the lstm_step_backward function that you just defined.     #
        #############################################################################
        N, T, H_time_2 = dh_all.shape #得到的误差是batch *T* 2H
        H = int(H_time_2 / 2)
        # N, T, H = dh_all.shape
        x, _, _, _, _, _, _, _, _, _ = self.gruParams[0]['cache'][0]
        D = x.shape[1]

        dh = dh_all[:,:,0:H]
        dih = dh_all[:,:,H:2*H]

        dweights = []

        for layer in range(self.layersNum - 1, -1, -1):

            dh_prevl = dh
            dih_prevl = dih

            DH = D if layer == 0 else H

            # 右向 得到前向传播保存的cache数组
            cache = self.gruParams[layer]['cache']

            dx = np.zeros((N, T, DH))
            dWzx = np.zeros((DH, 2 * H))
            dWzh = np.zeros((H, 2 * H))
            dbz = np.zeros((2 * H))

            dWax = np.zeros((DH, H))
            dWar = np.zeros((H, H))
            dba = np.zeros((H))

            dprev_h = np.zeros((N, H))

            # 左向
            icache = self.gruParams[layer]['icache']

            dix = np.zeros((N, T, DH))
            diWzx = np.zeros((DH, 2 * H))
            diWzh = np.zeros((H, 2 * H))
            dibz = np.zeros((2 * H))

            diWax = np.zeros((DH, H))
            diWar = np.zeros((H, H))
            diba = np.zeros((H))

            dprev_ih = np.zeros((N, H))

            for t in range(T - 1, -1, -1):
                # 右向
                dx[:, t, :], dprev_h, dWzx_t, dWzh_t, dbz_t, dWax_t, dWar_t, dba_t = self.gru_step_backward(
                    dh_prevl[:, t, :] + dprev_h,
                    cache[t])  # 注意此处的叠加
                dWzx += dWzx_t
                dWzh += dWzh_t
                dbz += dbz_t

                dWax += dWax_t
                dWar += dWar_t
                dba += dba_t

                dix[:, T-1-t, :], dprev_ih, diWzx_t, diWzh_t, dibz_t, diWax_t, diWar_t, diba_t = self.gru_step_backward(
                    dih_prevl[:, T- 1 - t, :] + dprev_ih,
                    icache[t])  # 注意此处的叠加，逆序

                diWzx += diWzx_t
                diWzh += diWzh_t
                dibz += dibz_t

                diWax += diWax_t
                diWar += diWar_t
                diba += diba_t

            dweight = (dWzx, dWzh, dbz, dWax, dWar, dba,
                       diWzx, diWzh, dibz, diWax, diWar, diba
                       #,dU, dV, dbc_final
                       )
            dweights.append(dweight)

            dh = dx
            dih = dix

        dh_t_all = dh + dih # 合并得到dx
        return dh_t_all, dweights

    def gru_step_forward(self, x, prev_h, Wzx, Wzh, bz, Wax, War, ba):
        """
        Forward pass for a single timestep of an GRU&BiGRU.

        The input data has dimension D, the hidden state has dimension H, and we use
        a minibatch size of N.

        Note that a sigmoid() function has already been provided for you in this file.

        Inputs:
        - x: Input data, of shape (N, D)
        - prev_h: Previous hidden state, of shape (N, H)
        - prev_c: previous cell state, of shape (N, H)
        - Wzx: Input-to-hidden weights, of shape (D, 4H)
        - Wh: Hidden-to-hidden weights, of shape (H, 4H)
        - b: Biases, of shape (4H,)

        Returns a tuple of:
        - next_h: Next hidden state, of shape (N, H)
        - next_c: Next cell state, of shape (N, H)
        - cache: Tuple of values needed for backward pass.
        """
        # next_h, cache = None, None
        #############################################################################
        # 首层，x(N,T,D), 向上变成xh(N,T,H)
        # 首层 Wx(D,H),   向上变成Wxh(H,H)
        #############################################################################
        H = prev_h.shape[1]
        # z_hat, of shape(N,4H)
        z_hat = Tools.matmul(x, Wzx) + Tools.matmul(prev_h, Wzh) + bz

        # of shape(N,H)
        r = Tools.sigmoid(z_hat[:, :H])
        z = Tools.sigmoid(z_hat[:, H:2 * H])

        a = Tools.matmul(x, Wax) + Tools.matmul(r * prev_h, War) + ba
        tanha = np.tanh(a)
        next_h = prev_h * (1. - z) + z * tanha

        # cache = (x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a, tanha)
        cache = (x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, tanha)

        return next_h, cache

    def gru_step_backward(self, dnext_h, cache):
        """
        Backward pass for a single timestep of an GRU&BiGRU.

        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        #############################################################################
        # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
        # the output value from the nonlinearity.                                   #
        #############################################################################
        # x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a, tanha = cache
        x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, tanha = cache

        # N, D = x.shape
        H = dnext_h.shape[1]

        # delta
        # tanha = np.tanh(a)
        dh = dnext_h
        da = dh * z * (1. - tanha ** 2)
        dh_prev_1 = dh * (1. - z)
        # dz_hat_2_n = dh * (tanha - prev_h) * (z * (1. - z))
        dz_hat_2 = (tanha - prev_h) * z * dh_prev_1
        # isSame1 = np.allclose(dz_hat_2,dz_hat_2_n)

        dhat_a = Tools.matmul(da, War.T)
        dr = dhat_a * prev_h

        dx_1 = Tools.matmul(da, Wax.T)
        dh_prev_2 = dhat_a * r  # da* Tools.matmul(r,War.T)
        dz_hat_1 = dr * (r * (1. - r))

        dz_hat = np.hstack((dz_hat_1, dz_hat_2))

        dx_2 = Tools.matmul(dz_hat_1, Wzx[:, :H].T)
        dh_prev_3 = Tools.matmul(dz_hat_1, Wzh[:, :H].T)

        dx_3 = Tools.matmul(dz_hat_2, Wzx[:, H:2 * H].T)
        dh_prev_4 = Tools.matmul(dz_hat_2, Wzh[:, H:2 * H].T)

        dprev_h = dh_prev_1 + dh_prev_2 + dh_prev_3 + dh_prev_4
        dx = dx_1 + dx_2 + dx_3

        dWax = Tools.matmul(x.T, da)
        dWar = Tools.matmul((r * prev_h).T, da)
        dba = np.sum(da, axis=0)

        dWzx = Tools.matmul(x.T, dz_hat)
        dWzh = Tools.matmul(prev_h.T, dz_hat)

        dbz = np.sum(dz_hat, axis=0)

        return dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba

    def gru_step_backward_succ(self, dnext_h, cache):
        """
        Backward pass for a single timestep of an GRU&BiGRU.

        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        #############################################################################
        # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
        # the output value from the nonlinearity.                                   #
        #############################################################################
        x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a = cache

        N, D = x.shape
        H = dnext_h.shape[1]

        # delta
        tanha = np.tanh(a)
        dh = dnext_h
        da = dh * z * (1. - tanha ** 2)
        dh_prev_1 = dh * (1. - z)
        # dz = dh*(tanha-prev_h)
        dz_hat_2 = dh * (tanha - prev_h) * (z * (1. - z))

        dhat_a = Tools.matmul(da, War.T)
        dr = dhat_a * prev_h

        dx_1 = Tools.matmul(da, Wax.T)
        dh_prev_2 = dhat_a * r  # da* Tools.matmul(r,War.T)
        dz_hat_1 = dr * (r * (1. - r))

        dz_hat = np.hstack((dz_hat_1, dz_hat_2))

        dx_2 = Tools.matmul(dz_hat_1, Wzx[:, :H].T)
        dh_prev_3 = Tools.matmul(dz_hat_1, Wzh[:, :H].T)

        dx_3 = Tools.matmul(dz_hat_2, Wzx[:, H:2 * H].T)
        dh_prev_4 = Tools.matmul(dz_hat_2, Wzh[:, H:2 * H].T)

        dprev_h = dh_prev_1 + dh_prev_2 + dh_prev_3 + dh_prev_4
        dx = dx_1 + dx_2 + dx_3

        dWax = Tools.matmul(x.T, da)
        dWar = Tools.matmul((r * prev_h).T, da)
        dba = np.sum(da, axis=0)

        dWzx = Tools.matmul(x.T, dz_hat)
        dWzh = Tools.matmul(prev_h.T, dz_hat)

        dbz = np.sum(dz_hat, axis=0)

        return dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba

