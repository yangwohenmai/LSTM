from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# prepare sequence
# 输入为1个样本，5个步长，1个特征值
length = 5
seq = array([i/float(length) for i in range(length)])
"""
[[[0. ]
  [0.2]
  [0.4]
  [0.6]
  [0.8]]]
"""
X = seq.reshape(1, 5, 1)
print(X)
# 输出为五个特征值的一个样本
"""
[[0.  0.2 0.4 0.6 0.8]]
"""
y = seq.reshape(1, 5)
print(y)
# define LSTM configuration
# create LSTM
model = Sequential()
# 输入类型为5个步长，1个特征值
model.add(LSTM(5, input_shape=(5, 1)))
model.add(Dense(5))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(X, y, epochs=500, batch_size=1, verbose=2)
# evaluate
result = model.predict(X, batch_size=1, verbose=0)
for value in result[0,:]:
	print('%.1f' % value)