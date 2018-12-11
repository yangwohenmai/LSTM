from pandas import DataFrame
# convert sequence to x/y pairs ready for use with an LSTM
def to_lstm_dataset(sequence, n_unique):
    # one hot encode
    encoded = encode(sequence, n_unique)
    # 建立输入输出对应关系，y[i] = x[i+1]
    X,y = to_xy_pairs(encoded)
    """
    [[0.0, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]]
    """
    print(X)
    """
    [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]]
    """
    print(y)
    # 数据X，y处理成LSTM可以处理的矩阵格式
    dfX, dfy = DataFrame(X), DataFrame(y)
    """
         0    1    2    3    4
    0  0.0  0.0  0.0  1.0  0.0
    1  1.0  0.0  0.0  0.0  0.0
    2  0.0  1.0  0.0  0.0  0.0
    3  0.0  0.0  1.0  0.0  0.0
    """
    print(dfX)
    """
         0    1    2    3    4
    0  1.0  0.0  0.0  0.0  0.0
    1  0.0  1.0  0.0  0.0  0.0
    2  0.0  0.0  1.0  0.0  0.0
    3  0.0  0.0  0.0  1.0  0.0
    """
    print(dfy)
    # 提取矩阵中的数据，变成二维数据格式
    """
    [[0. 0. 0. 1. 0.]
     [1. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0.]
     [0. 0. 1. 0. 0.]]
    """
    lstmX = dfX.values
    print(lstmX)
    # lstmX.reshape(4, 1, 5),格式化成四个样本，每个样本一个步长，每个样本5个特征值。
    # lstmX.shape[0]是行数，用来作为样本数量，lstmX.shape[1]是列数，用来作为特征值。
    """
    [[[0. 0. 0. 1. 0.]]

     [[1. 0. 0. 0. 0.]]

     [[0. 1. 0. 0. 0.]]

     [[0. 0. 1. 0. 0.]]]
    """
    lstmX = lstmX.reshape(lstmX.shape[0], 1, lstmX.shape[1])
    print(lstmX)
    lstmY = dfy.values
    """
    [[1. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0.]
     [0. 0. 1. 0. 0.]
     [0. 0. 0. 1. 0.]]
    """
    print(lstmY)
    return lstmX, lstmY

# binary encode an input pattern, return a list of binary vectors
def encode(pattern, n_unique):
    encoded = list()
    for value in pattern:
        row = [0.0 for x in range(n_unique)]
        row[value] = 1.0
        encoded.append(row)
    return encoded

def to_xy_pairs(encoded):
    X,y = list(),list()
    for i in range(1, len(encoded)):
        X.append(encoded[i-1])
        y.append(encoded[i])
    return X, y

seq1 = [3, 0, 1, 2, 3]
seq2 = [4, 0, 1, 2, 4]
n_unique = len(set(seq1 + seq2))

seq1X, seq1Y = to_lstm_dataset(seq1, n_unique)
seq2X, seq2Y = to_lstm_dataset(seq2, n_unique)
# 输出结果
"""
[[[0. 0. 0. 1. 0.]]

 [[1. 0. 0. 0. 0.]]

 [[0. 1. 0. 0. 0.]]

 [[0. 0. 1. 0. 0.]]]
"""
print(seq1X)
"""
[[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]]
"""
print(seq1Y)
"""
[[[0. 0. 0. 0. 1.]]

 [[1. 0. 0. 0. 0.]]

 [[0. 1. 0. 0. 0.]]

 [[0. 0. 1. 0. 0.]]]
"""
print(seq2X)
"""
[[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 1.]]
"""
print(seq2Y)