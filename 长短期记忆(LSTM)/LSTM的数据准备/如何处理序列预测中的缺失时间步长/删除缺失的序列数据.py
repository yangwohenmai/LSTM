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
	sequence = array(sequence)
	# 格式化成有序矩阵
	df = DataFrame(sequence)
	df = concat([df.shift(1), df], axis=1)
	# 删除空值的数据
	df.dropna(inplace=True)
	values = df.values
	# 指定输入和输出数据
	X, y = values, values[:, 0]
	return X, y

# 生成序列
n_timesteps = 10
X, y = generate_data(n_timesteps)
# print sequence
for i in range(len(X)):
	print(X[i], '=>', y[i])