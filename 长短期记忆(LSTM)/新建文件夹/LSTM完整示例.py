from pandas import DataFrame
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# 数据转码成二进制方法
def encode(pattern, n_unique):
	encoded = list()
	for value in pattern:
		row = [0.0 for x in range(n_unique)]
		row[value] = 1.0
		encoded.append(row)
	return encoded

# create input/output pairs of encoded vectors, returns X, y
def to_xy_pairs(encoded):
	X,y = list(),list()
	for i in range(1, len(encoded)):
		X.append(encoded[i-1])
		y.append(encoded[i])
	return X, y

# convert sequence to x/y pairs ready for use with an LSTM
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
# 将序列转换为所需的数据格式
n_unique = len(set(seq1 + seq2))
seq1X, seq1Y = to_lstm_dataset(seq1, n_unique)
seq2X, seq2Y = to_lstm_dataset(seq2, n_unique)
# 定义 LSTM 的配置
n_neurons = 20 #网络节点数量
n_batch = 1 #样本
n_epoch = 650 #训练周期
n_features = n_unique #特征值
# 创建 LSTM网络
model = Sequential()
model.add(LSTM(n_neurons, batch_input_shape=(n_batch, 1, n_features), stateful=True))
model.add(Dense(n_unique, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
# 训练 LSTM网络
for i in range(n_epoch):
	model.fit(seq1X, seq1Y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
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