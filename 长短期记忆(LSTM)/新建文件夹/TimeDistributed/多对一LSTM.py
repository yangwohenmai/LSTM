from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# prepare sequence
# 输入为一个样本，五个步长，一个特征值
# 输出为五个特征值的一个样本
length = 5
seq = array([i/float(length) for i in range(length)])
X = seq.reshape(1, 5, 1)
y = seq.reshape(1, 5)
# define LSTM configuration
n_neurons = length
n_batch = 1
n_epoch = 500
# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(5, 1)))
model.add(Dense(5))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result[0,:]:
	print('%.1f' % value)