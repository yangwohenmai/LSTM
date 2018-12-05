from random import random
from numpy import array
from pandas import concat
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Masking

# 生成一系列随机值
def generate_sequence(n_timesteps):
	return [random() for _ in range(n_timesteps)]

# generate data for the lstm
def generate_data(n_timesteps):
	# generate sequence
	sequence = generate_sequence(n_timesteps)
	sequence = array(sequence)
	# create lag
	df = DataFrame(sequence)
	df = concat([df.shift(1), df], axis=1)
	#将NAN值替换为-1
	df.fillna(-1, inplace=True)
	values = df.values
	# specify input and output data
	X, y = values, values[:, 1]
	# reshape
	X = X.reshape(len(X), 2, 1)
	y = y.reshape(len(y), 1)
	return X, y

n_timesteps = 10
# 定义网络
model = Sequential()
# mask_value=-1使得网络对-1的值进行忽略，不进行学习
model.add(Masking(mask_value=-1, input_shape=(2, 1)))
model.add(LSTM(5))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit model
for i in range(500):
	X, y = generate_data(n_timesteps)
	model.fit(X, y, epochs=1, batch_size=1, verbose=2)
# 评估数据模型
X, y = generate_data(n_timesteps)
yhat = model.predict(X)
for i in range(len(X)):
	print('Expected', y[i,0], 'Predicted', yhat[i,0])