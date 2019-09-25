from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
# prepare sequence
length = 5
seq = array([i/float(length) for i in range(length)])
# 输入为1个样本，5个步长，1个特征值
X = seq.reshape(1, 5, 1)
# 输出为1个样本，5个步长，1个特征值
y = seq.reshape(1, 5, 1)
# define LSTM configuration
# create LSTM
model = Sequential()
# 输入类型为5个步长和1个特征值，return_sequences=True返回整个序列
model.add(LSTM(5, input_shape=(5, 1), return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(X, y, epochs=1000, batch_size=1, verbose=2)
# evaluate
result = model.predict(X, batch_size=1, verbose=0)
for value in result[0,:,0]:
	print('%.1f' % value)