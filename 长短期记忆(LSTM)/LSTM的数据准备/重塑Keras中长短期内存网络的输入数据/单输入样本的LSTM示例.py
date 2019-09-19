from numpy import array
"""
https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
"""
# 将这个数字序列定义为NumPy数组
data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# 我们可以使用NumPy数组上的reshape（）函数将这个一维数组重新整形为一个三维数组，每个时间步长有1个样本，10个时间步长和1个特征(列)
data = data.reshape((1, 10, 1))
print(data.shape)
print(data)
