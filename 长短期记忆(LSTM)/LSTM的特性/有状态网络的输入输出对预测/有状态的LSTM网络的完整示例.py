from pandas import DataFrame
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# 数据转码成one-hot编码
def encode(pattern, n_unique):
	encoded = list()
	for value in pattern:
		row = [0.0 for x in range(n_unique)]
		row[value] = 1.0
		encoded.append(row)
	return encoded

# 创建一个输入输出数据映射
def to_xy_pairs(encoded):
	X,y = list(),list()
	for i in range(1, len(encoded)):
		X.append(encoded[i-1])
		y.append(encoded[i])
	return X, y

# 创建一个输入输出数据映射的序列
def to_lstm_dataset(sequence, n_unique):
	# one hot 编码
	encoded = encode(sequence, n_unique)
	# 转换数据成输入输出对
	X,y = to_xy_pairs(encoded)
	# 转换数据成LSTM可识别的矩阵格式
	dfX, dfy = DataFrame(X), DataFrame(y)
	lstmX = dfX.values
	lstmX = lstmX.reshape(lstmX.shape[0], 1, lstmX.shape[1])
	lstmY = dfy.values
	return lstmX, lstmY

# define sequences
seq1 = [3, 0, 1, 2, 3]
seq2 = [4, 0, 1, 2, 4]
# 获取序列的特征值数量
n_unique = len(set(seq1 + seq2))
# 创建一批用于训练的输入输出数据序列
seq1X, seq1Y = to_lstm_dataset(seq1, n_unique)
seq2X, seq2Y = to_lstm_dataset(seq2, n_unique)
# 定义 LSTM 的配置
n_neurons = 20 #网络节点数量
n_batch = 1 #样本
n_epoch = 650 #训练周期
n_features = n_unique #特征值
# 创建LSTM网络
model = Sequential()
# 网络中间层节点20个，每次输入样本1个，步长为1，特征值为5，stateful=True表明记录当前模型状态，作为为下一次训练网络时的前置状态
model.add(LSTM(20, batch_input_shape=(n_batch, 1, n_features), stateful=True))
# 激活函数为sigmoid函数，n_unique为输入数据维度/输入特征值数量
model.add(Dense(n_unique, activation='sigmoid'))
# loss为交叉熵损失函数，优化器是adam
model.compile(loss='binary_crossentropy', optimizer='adam')
# 训练LSTM网络，本文的网络训练方法是，每次网络训练一个周期并记住当前网络状态，并将当前网络状态作为下次网络训练的初始状态带入
for i in range(650):
	# 每次训练一个样本，一个训练周期，verbose=1表示打印网络执行状态，shuffle=False不打乱训练数据的顺序
	model.fit(seq1X, seq1Y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
	# 我们对这批数据训练650次，可以理解为有650条数据，而这650条数据都是独立的，之间并没有顺序关联，所以每次都重置网络状态
    # 如果是将一个完整的长序列拆分成650份短序列，则这650条数据之间是有关联的，每次循环可不重置网络状态，使状态在每次训练中传递下去
	model.reset_states()
	model.fit(seq2X, seq2Y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
	model.reset_states()

# 测试 LSTM对“数列1”预测
print('Sequence 1')
result = model.predict_classes(seq1X, batch_size=n_batch, verbose=0)
model.reset_states()
for i in range(len(result)):
	print('X=%.1f y=%.1f, yhat=%.1f' % (seq1[i], seq1[i+1], result[i]))

# 测试 LSTM对“数列2”预测
print('Sequence 2')
result = model.predict_classes(seq2X, batch_size=n_batch, verbose=0)
model.reset_states()
for i in range(len(result)):
	print('X=%.1f y=%.1f, yhat=%.1f' % (seq2[i], seq2[i+1], result[i]))