from random import randint
from numpy import array
from numpy import argmax
from pandas import DataFrame
from pandas import concat

# generate a sequence of random numbers in [0, 49]
def generate_sequence(length=25):
	return [randint(0, 99) for _ in range(length)]

# 生成一个one hot encode 序列
def one_hot_encode(sequence, n_unique=100):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)

# 解码one hot encoded 的字符串
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# 将编码序列转换为监督学习
def to_supervised(sequence, n_in, n_out):
	# 创建序列的滞后副本
	df = DataFrame(sequence)
	df = concat([df.shift(n_in-i-1) for i in range(n_in)], axis=1)
	# 删除缺失数据的行
	df.dropna(inplace=True)
	# 指定输入和输出对的列
	values = df.values
	width = sequence.shape[1]
	X = values.reshape(len(values), n_in, width)
	y = values[:, 0:(n_out*width)].reshape(len(values), n_out, width)
	return X, y

# 生成随机序列
sequence = generate_sequence()
print(sequence)
# 转码成one hot encode
encoded = one_hot_encode(sequence)
print(encoded)
# 构造一个X y 键值对
X,y = to_supervised(encoded, 5, 3)
# 构造键值对
for i in range(len(X)):
	print(one_hot_decode(X[i]), '=>', one_hot_decode(y[i]))