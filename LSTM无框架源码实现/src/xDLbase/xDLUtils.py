"""
Tool Helper
"""

import numpy as np
import numba

class Tools:
    # padding before cross-correlation and pooling
    @staticmethod
    def padding(x, pad, data_type):

        size_x = x.shape[2]  # 输入矩阵尺寸
        size = size_x + pad * 2  # padding后尺寸
        if x.ndim == 4:  # 每个元素是3维的，x的0维是mini-batch
            # 初始化同维全0矩阵
            padding = np.zeros((x.shape[0], x.shape[1], size, size), dtype=data_type)
            # 中间以x填充
            padding[:, :, pad: pad + size_x, pad: pad + size_x] = x

        elif x.ndim == 3:  # 每个元素是2维的
            padding = np.zeros((x.shape[0], size, size), dtype=data_type)
            padding[:, pad: pad + size_x, pad: pad + size_x] = x

        return padding

    # 执行环境内存充裕blas方法较快
    # 否则使用jit后的np.matmul方法
    #@numba.jit
    def matmul(a, b):
        return np.matmul(a, b)

    # 输出层结果转换为标准化概率分布，
    # 入参为原始线性模型输出y ，N*K矩阵，
    # 输出矩阵规格不变
    @staticmethod
    def softmax(y):
        # 对每一行：所有元素减去该行的最大的元素,避免exp溢出,得到1*N矩阵,
        max_y = np.max(y, axis=1)
        # 极大值重构为N * 1 数组
        max_y.shape = (-1, 1)
        # 每列都减去该列最大值
        y1 = y - max_y
        # 计算exp
        exp_y = np.exp(y1)
        # 按行求和，得1*N 累加和数组
        sigma_y = np.sum(exp_y, axis=1)
        # 累加和reshape为N*1 数组
        sigma_y.shape = (-1, 1)
        # 计算softmax得到N*K矩阵
        softmax_y = exp_y / sigma_y

        return softmax_y

    # 交叉熵损失函数
    # 限制上界避免除零错
    @staticmethod
    def crossEntropy(y,y_):
        n = len(y)
        return np.sum(-np.log(np.clip(y[range(n), y_],1e-10,None,None)))/n

    @staticmethod
    def crossEntropylogit(y,y_):
        return -np.log(np.clip(y[range(len(y)), y_],1e-10,None,None))

    # def crossEntropy(y,y_,eps):
    #     return -np.log(np.clip(y[range(len(y)), y_],eps,None,None))

    # 平方误差损失
    @staticmethod
    def mse(y, y_):
        return np.mean((y - y_) ** 2, axis=-1) / 2

    # sigmoid
    @staticmethod
    def sigmoid(x):
        """
        A numerically stable version of the logistic sigmoid function.
        """
        pos_mask = (x >= 0)
        neg_mask = (x < 0)
        z = np.zeros_like(x)
        z[pos_mask] = np.exp(-x[pos_mask])
        z[neg_mask] = np.exp(x[neg_mask])
        top = np.ones_like(x)
        top[neg_mask] = z[neg_mask]
        return top / (1 + z)

    @staticmethod
    def bp4sigmoid(y):
        """
        A numerically stable version of the logistic sigmoid function.
        """
        return y * (1 - y)

    @staticmethod
    def bp4tanh(y):
        """
        A numerically stable version of the logistic sigmoid function.
        """
        return 1 - y**2


    def dropout4rnn(x, p):
        """
        input:
            x:input batch*T*D
            p: reserve percentage  (0,1], 1 means no dropout
        return
            hat_x : dropout on T*D dimention , same shape with x
            mask  : mask matrix for bp
        """
        if p <= 0. or p > 1:
            raise ValueError('Dropout reserve p must be in interval (0, 1].')

        if p == 1 :
            return x,None

        mask=np.random.binomial(1,p,x[0].shape)
        hat_x = (x * mask )/p

        return hat_x,mask
    
    # 此方法本质是获得一个随机初始化的矩阵
    # 函数：np.linalg.svd(a,full_matrices=1,compute_uv=1)。
    # 参数：
    # a是一个形如(M,N)矩阵
    # full_matrices的取值是为0或者1，默认值为1，这时u的大小为(M,M)，v的大小为(N,N) 。
    # 否则u的大小为(M,K)，v的大小为(K,N) ，K=min(M,N)。
    # compute_uv的取值是为0或者1，默认值为1，表示计算u,s,v。为0的时候只计算s。
    # 返回值：
    # 总共有三个返回值u,s,v
    # u大小为(M,M)，s大小为(M,N)，v大小为(N,N)。
    # -------
    # A = u*s*v
    # 其中s是对矩阵a的奇异值分解。s除了对角元素不为0，其他元素都为0，并且对角元素从大到小排列。
    # s中有n个奇异值，一般排在后面的比较接近0，所以仅保留比较大的r个奇异值。
    # 矩阵的奇异值分解可将一个大矩阵分解成三个小矩阵，减少了存储空间同时也便于计算
    @staticmethod
    def initOrthogonal(shape,initRng,dType):
        reShape =  (shape[0], np.prod(shape[1:]))
        # 在区间范围内按reShape形状取样
        x = np.random.uniform(-1 * initRng, initRng, reShape).astype(dType)
        # x = np.random.normal(-1 * initRng, initRng, reShape).astype(dType)
        # x = np.random.normal(0, 1, reShape).astype(dType)
        # 矩阵奇异值分解
        u,_,vt= np.linalg.svd(x,full_matrices =False)
        w = u if u.shape == reShape else vt
        w = w.reshape(shape)
        return w
