from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# prepare sequence
length = 5
seq = array([i/float(length) for i in range(length)])
# 定义输入为：5个样本，一个步长，一个特征值
"""
[[[0. ]]

 [[0.2]]

 [[0.4]]

 [[0.6]]

 [[0.8]]]
"""
X = seq.reshape(5, 1, 1)
print(X)
# 定义输出为：5个样本，每个样本1个特征值
"""
[[0. ]
 [0.2]
 [0.4]
 [0.6]
 [0.8]]
"""
y = seq.reshape(5, 1)
print(y)
# define LSTM configuration
n_neurons = 5
n_batch = 5
# create LSTM
model = Sequential()
# 输入类型为1个步长和1个特征值
model.add(LSTM(5, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(X, y, epochs=1000, batch_size=n_batch, verbose=2)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result:
	print('%.1f' % value)