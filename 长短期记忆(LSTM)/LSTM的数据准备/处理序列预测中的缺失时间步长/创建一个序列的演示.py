from random import random
from numpy import array
from pandas import concat
from pandas import DataFrame

# 生成一系列随机值
def generate_sequence(n_timesteps):
	return [random() for _ in range(n_timesteps)]

# 生成lstm的数据
def generate_data(n_timesteps):
    # 生成随机序列
    sequence = generate_sequence(n_timesteps)
    print(sequence)
    sequence = array(sequence)
    print(sequence)
    # 格式化成有序矩阵
    df = DataFrame(sequence)
    print(df)
    # 变成两两一组的有序数据矩阵，创建可用于在时间步长之前以代表观测序列的移位版
    df = concat([df.shift(1), df], axis=1)
    print(df)
    values = df.values
    print(values)
    # 指定输入和输出数据
    X, y = values, values[:, 0]
    print(y)
    return X, y

# 生成序列
n_timesteps = 10
X, y = generate_data(n_timesteps)
# 打印序列
for i in range(n_timesteps):
    print(X[i], '=>', y[i])