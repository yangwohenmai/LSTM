"""
Main Function
"""

from unittest import result
import numpy as np
import logging.config
import random, time
import matplotlib.pyplot as plt

import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(rootPath+'/xDLbase')
sys.path.append(rootPath+'/xutils')

from xDLbase.xview import *
from xDLbase.fc import *
from xDLbase.rnn import *
from xDLbase.optimizers import *
from xDLbase.activators import *
from xDLbase.session import *

# create logger
exec_abs = os.getcwd()
log_conf = exec_abs + '/config/logging.conf'
logging.config.fileConfig(log_conf)
logger = logging.getLogger('main')

# 持久化配置
trace_file_path = 'D:/0tmp/'
exec_name = os.path.basename(__file__)
trace_file = trace_file_path + exec_name + ".data"

# General params
class Params:

    EPOCH_NUM = 30  # EPOCH
    MINI_BATCH_SIZE = 32  # batch_size
    ITERATION = 1  # 每batch训练轮数
    # LEARNING_RATE = 0.005  # Vanilla E5:loss 0.0014, 好于AdaDelta的0.0021
    LEARNING_RATE = 0.01  # LSTM
    # LEARNING_RATE = 0.002  # GRU
    # LEARNING_RATE = 0.1  # BiLSTM
    # LEARNING_RATE = 0.1  # BiGRU
    # LEARNING_RATE = 0.05  # BiGRU+ReLU
    # VAL_FREQ = 30  # val per how many batches
    VAL_FREQ = 5  # val per how many batches
    # LOG_FREQ = 10  # log per how many batches
    LOG_FREQ = 1  # log per how many batches


    HIDDEN_SIZE = 30  # LSTM中隐藏节点的个数,每个时间节点上的隐藏节点的个数，是w的维度.
    # RNN/LSTM/GRU每个层次的的时间节点个数，有输入数据的元素个数确定。
    NUM_LAYERS = 2  # RNN/LSTM的层数。
    # 设置缺省数值类型
    DTYPE_DEFAULT = np.float32
    INIT_W = 0.01  # 权重矩阵初始化参数

    DROPOUT_R_RATE = 1 # dropout比率
    TIMESTEPS = 1  # 循环神经网络的训练序列长度。
    PRED_STEPS = TIMESTEPS  # 预测序列长度
    TRAINING_STEPS = 10000  # 训练轮数。
    TRAINING_EXAMPLES = 10000  # 训练数据个数。
    TESTING_EXAMPLES = 1000  # 测试数据个数。
    SAMPLE_GAP = 0.01  # 采样间隔。
    VALIDATION_CAPACITY = TESTING_EXAMPLES-TIMESTEPS  # 验证集大小
    TYPE_K = 2  # 分类类别

    # 持久化开关
    TRACE_FLAG = False
    # loss曲线开关
    SHOW_LOSS_CURVE = True

    # Optimizer params
    BETA1 = 0.9
    BETA2 = 0.999
    EPS = 1e-8
    EPS2 = 1e-10
    REG_PARA = 0.5  # 正则化乘数
    LAMDA = 1e-4  # 正则化系数lamda
    INIT_RNG=1e-4

    # 并行度
    # TASK_NUM_MAX = 3
    # 任务池
    # g_pool = ProcessPoolExecutor(max_workers=TASK_NUM_MAX)


# data loading
class SeqData(object):

    def __init__(self, dataType):
        self.dataType = dataType
        # 程序生成数据集
        self.x, self.y,self.x_v, self.y_v = self.initData()
        # 训练集样本数据索引表
        self.sample_range = [i for i in range(len(self.y))]
        # 训练集样本数据索引表
        self.sample_range_v = [i for i in range(len(self.y_v))]

    def initData(self):
        #return self.SinData()
        #return self.SumData()
        return self.ClassifyData()
    
    # 求和结果分类，x1+x2>60
    def ClassifyData(self):
        xArray = []
        yArray = []
        for _ in range(Params.TRAINING_EXAMPLES + Params.TESTING_EXAMPLES):
            num1 = np.random.randint(0, 50)
            num2 = np.random.randint(0, 50)
            sum = num1 + num2
            xArray.append([num1, num2])
            if sum >= 60:
                yArray.append(1)
            else:
                yArray.append(0)
        # 监督学习数据 n*[X1, X2] -> n*[y]  <=> X.shape(sample, 1 , 2) -> Y.shape(sample, 1, 1)
        trainX = np.array(xArray[:Params.TRAINING_EXAMPLES]).reshape(Params.TRAINING_EXAMPLES, 1, 2)
        trainY = np.array(yArray[:Params.TRAINING_EXAMPLES])

        testX = np.array(xArray[Params.TRAINING_EXAMPLES:]).reshape(Params.TESTING_EXAMPLES, 1, 2)
        testY = np.array(yArray[Params.TRAINING_EXAMPLES:])

        return trainX, trainY,testX,testY

    # 用求和公式生成训练和测试数据集合。
    def SumData(self):
        xArray = []
        yArray = []
        for _ in range(Params.TRAINING_EXAMPLES + Params.TESTING_EXAMPLES):
            num1 = np.random.randint(0, 50)
            num2 = np.random.randint(0, 50)
            sum  = num1 + num2
            xArray.append([num1, num2])
            yArray.append([sum])
        # 监督学习数据 n*[X1, X2] -> n*[y]  <=> X.shape(sample, 1 , 2) -> Y.shape(sample, 1, 1)
        trainX = np.array(xArray[:Params.TRAINING_EXAMPLES]).reshape(Params.TRAINING_EXAMPLES, 1, 2)
        trainY = np.array(yArray[:Params.TRAINING_EXAMPLES]).reshape(Params.TRAINING_EXAMPLES, 1, 1)

        testX = np.array(xArray[Params.TRAINING_EXAMPLES:]).reshape(Params.TESTING_EXAMPLES, 1, 2)
        testY = np.array(yArray[Params.TRAINING_EXAMPLES:]).reshape(Params.TESTING_EXAMPLES, 1, 1)

        return trainX, trainY,testX,testY
    
    # 用正弦函数生成训练和测试数据集合。
    def SinData(self):
        # (1w+10)*0.01 测试集起始位置
        test_start = (Params.TRAINING_EXAMPLES + Params.TIMESTEPS) * Params.SAMPLE_GAP
        #  (1w+10)*0.01 + (1w+10)*0.01 测试集结束位置
        test_end = test_start + (Params.TESTING_EXAMPLES + Params.TIMESTEPS) * Params.SAMPLE_GAP

        # np.linspace 生成等差数列(start-首项,stop-尾项,number-项数,endpoint-末项是否包含在内，默认true包含)
        # curve 内调用非线性组合函数，将等差数列转换成非线性序列
        train_X, train_y = self.generate_data(self.curve(np.linspace(0, test_start, Params.TRAINING_EXAMPLES + Params.TIMESTEPS, dtype=Params.DTYPE_DEFAULT)))
        test_X, test_y = self.generate_data(self.curve(np.linspace(test_start, test_end, Params.TESTING_EXAMPLES + Params.TIMESTEPS, dtype=Params.DTYPE_DEFAULT)))
        return train_X, train_y,test_X,test_y

    # 构建输入和输出的数据集 X y
    def generate_data(self,seq):
        X = []
        y = []
        # 序列的第i项和后面的TIMESTEPS-1项合在一起的10个点序列作为输入；第i + TIMESTEPS项这个点作为输出。
        # 即用curve函数前面从i开始的TIMESTEPS=10个点的信息，预测第i + TIMESTEPS这个点的函数值。（改）
        # 即用curve函数前面从i开始的TIMESTEPS=10个点的信息[i, i+TIMESTEPS-1]，预测第[i+TIMESTEPS, i+TIMESTEPS+PRED_STEPS]这个区间的函数值。
        # 一共生成TRAINING_EXAMPLES = 1w对 序列、值数据对用于训练，同理生成验证数据
        # 交换维度为N,T,D :(N,10,1)->(N,1,10)
        # for i in range(len(seq) - Params.TIMESTEPS):
        for i in range(len(seq) - Params.TIMESTEPS-Params.PRED_STEPS):
            X.append([seq[i: i + Params.TIMESTEPS]])
            #y.append([seq[i + Params.TIMESTEPS]])
            y.append([seq[i + Params.TIMESTEPS:i + Params.TIMESTEPS + Params.PRED_STEPS]])
        #return np.swapaxes(np.array(X, dtype=self.dataType),1,2), np.array(y, dtype=self.dataType)
        return np.swapaxes(np.array(X, dtype=self.dataType),1,2), np.swapaxes(np.array(y, dtype=self.dataType),1,2)

    def curve(self,x):
        #return np.sin(np.pi * x / 50) + np.cos(np.pi * x / 50) + np.sin(np.pi * x / 25)
        return np.sin(np.pi * x / 3)
        #return np.sin(np.pi * x / 3.) + np.cos(np.pi * x / 3.) + np.sin(np.pi * x / 1.5)++ np.random.uniform(-0.05,0.05,len(x))
        
    # 对训练样本序号按照miniBatchSize尺寸随机分成(sample_range%miniBatchSize)组
    # 剩余不足miniBatchSize的单独成一组
    # 功能类似shuffle = True
    def getTrainRanges(self, miniBatchSize):
        rangeAll = self.sample_range
        random.shuffle(rangeAll)
        rngs = [rangeAll[i:i + miniBatchSize] for i in range(0, len(rangeAll), miniBatchSize)]
        return rngs

    # 根据传入的训练样本序号，获取对应的输入x和输出y
    def getTrainDataByRng(self, rng):
        xs = np.array([self.x[sample] for sample in rng], self.dataType)
        values = np.array([self.y[sample] for sample in rng])
        return xs, values

    # 获取验证样本,不打乱，用于显示连续曲线
    def getValData(self, valCapacity):

        samples_v = [i for i in range(valCapacity)]
        #  验证输入 N*28*28
        x_v = np.array([self.x_v[sample_v] for sample_v in samples_v], dtype=self.dataType)
        #  正确类别 1*K
        y_v = np.array([self.y_v[sample_v] for sample_v in samples_v])

        return x_v, y_v

def main_rnn():
    logger.info('start..')
    # 初始化
    try:
        os.remove(trace_file)
    except FileNotFoundError:
        pass

    # if (True == Params.SHOW_LOSS_CURVE):
    # 配置训练结果展示图的参数
    view = ResultView(Params.EPOCH_NUM,
                      ['train_loss', 'val_loss', 'train_acc', 'val_acc'],
                      ['k', 'r', 'g', 'b'],
                      ['Iteration', 'Loss', 'Accuracy'],
                      Params.DTYPE_DEFAULT)
    s_t = 0

    # 构建监督学习数据，维度为N,T,D :(N,10,1)->(N,1,10)
    seqData = SeqData(Params.DTYPE_DEFAULT)


    # 定义网络结构，优化器参数，支持各层使用不同的优化器。
    # optmParamsRnn1 = (Params.BETA1, Params.BETA2, Params.EPS)
    # optimizer = AdagradOptimizer

    # optmParamsRnn1 = (Params.BETA1,Params.EPS)
    # optimizer = AdaDeltaOptimizer

    optmParamsRnn1 = (Params.BETA1, Params.BETA2, Params.EPS)
    optimizer = AdamOptimizer

    # rnn
    # rnn1 = RnnLayer('rnn1',Params.MINI_BATCH_SIZE,Params.HIDDEN_SIZE,3,optimizer,optmParamsRnn1,Params.DROPOUT_R_RATE,Params.DTYPE_DEFAULT,Params.INIT_RNG)

    #LSTM
    rnn1 = LSTMLayer('lstm1',Params.MINI_BATCH_SIZE,Params.HIDDEN_SIZE,3,optimizer,optmParamsRnn1,Params.DROPOUT_R_RATE,Params.DTYPE_DEFAULT,Params.INIT_RNG)

    #BiLSTM
    # rnn1 = BiLSTMLayer('Bilstm1',Params.MINI_BATCH_SIZE,Params.HIDDEN_SIZE,3,optimizer,optmParamsRnn1,Params.DROPOUT_R_RATE,Params.DTYPE_DEFAULT,Params.INIT_RNG)

    #GRU
    #rnn1 = GRULayer('gru1',Params.MINI_BATCH_SIZE,Params.HIDDEN_SIZE,3,optimizer,optmParamsRnn1,Params.DROPOUT_R_RATE,Params.DTYPE_DEFAULT,Params.INIT_RNG)

    #BiGRU
    # rnn1 = BiGRULayer('bigru1',Params.MINI_BATCH_SIZE,Params.HIDDEN_SIZE,3,optimizer,optmParamsRnn1,Params.DROPOUT_R_RATE,Params.DTYPE_DEFAULT,Params.INIT_RNG)

    # 全连接层优化器
    optmParamsFc1=(Params.BETA1, Params.BETA2, Params.EPS)
    # RNN输出全部T个节点，FC层先把B,T,H拉伸成N,T*H, 再用仿射变换的W T*H,D 得到 N*D输出。
    # TIMESTEPS*HIDDEN_SIZE作为输入尺寸，PRED_STEPS表示预测值长度（步长），作为输出尺寸
    fc1 = FCLayer(Params.MINI_BATCH_SIZE, Params.TIMESTEPS*Params.HIDDEN_SIZE, Params.TYPE_K, NoAct, AdamOptimizer, optmParamsFc1,True,Params.DTYPE_DEFAULT,Params.INIT_W)
    # 拼接网络层
    seqLayers = [rnn1,fc1]

    # 生成训练模型实例
    #sess = Session(seqLayers,MseLoss)
    sess = Session(seqLayers,SoftmaxCrossEntropyLoss)
    #lrt = Params.LEARNING_RATE
    #lrt = 1
    # 开始训练过程，训练EPOCH_NUM个epoch
    iter = 0
    for epoch in range(Params.EPOCH_NUM):
        # 获取当前epoch使用的learing rate
        # for key in Params.DIC_L_RATE.keys():
        #     if (epoch + 1) < key:
        #         break
        #     lrt = Params.DIC_L_RATE[key]
        lrt = Params.LEARNING_RATE
        
        #if loss_v1<1000* lrt:
        #    lrt  = lrt /10
        
        # logger.info("epoch %2d, learning_rate= %.8f" % (epoch, lrt))
        # 随机打乱训练样本顺序，功能类似 shuffle=True
        dataRngs = seqData.getTrainRanges(Params.MINI_BATCH_SIZE)

        # 当前epoch中，对n组sample进行训练，每个sample包含BATCH_SIZE个样本
        for batch in range(len(dataRngs)):
            start = time.time()
            # 根据getTrainRanges打乱后的序号，取训练数据
            x, y_ = seqData.getTrainDataByRng(dataRngs[batch])
            # 训练模型，输出序列，只需要比较第0维；y.shape->(32,10,1),y_[:,:,0].shape->(32,10),相当于只是对y_进行降维，没有改变数据
            #_, loss_t = sess.train_steps(x, y_[:,:,0], lrt)
            # x(32, 1, 2) y_(32,)
            _, loss_t = sess.train_steps(x, y_, lrt)
            iter += 1


            if (batch % Params.LOG_FREQ == 0):  # 若干个batch show一次日志
                logger.info("epoch %2d-%3d, loss= %.8f st[%.1f]" % (epoch, batch, loss_t,  s_t))

            # 使用随机验证样本验证结果
            if (batch % Params.VAL_FREQ == 0 and (batch + epoch) > 0):
                #x_v, y_v = seqData.getValData(Params.VALIDATION_CAPACITY)
                x_v, y_v = seqData.getValData(Params.TESTING_EXAMPLES)
                # 多个出数值的序列，只需要比较0维
                #y, loss_v,_ = sess.validation(x_v, y_v[:,:,0])
                y, loss_v,_,result = sess.validation(x_v, y_v)

                logger.info('epoch %2d-%3d, loss=%f, loss_v=%f' % (epoch, batch, loss_t, loss_v))

                if (True == Params.SHOW_LOSS_CURVE):
                    # view.addData(fc1.optimizerObj.Iter,
                    view.addData(iter, loss_t, loss_v, 0, 0)
            s_t = time.time() - start

    logger.info('session end')
    x_v, y_v = seqData.getValData(Params.VALIDATION_CAPACITY)
    y, loss_v,_,result = sess.validation(x_v, y_v[:,:,0])
    plt.figure()
    plt.plot(y[:,0],linewidth=1.5, ls=':',label='predictions')
    plt.plot(y_v[:,0,0],linewidth=0.5, ls='-', label='real_curve')
    plt.legend()
    plt.show()
    view.show()

if __name__ == '__main__':
    main_rnn()
    input()
