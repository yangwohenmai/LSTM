from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
"""
TimeDistributed这个封装器将一个层应用于输入的每个时间片,输入至少为 3D，且第一个维度应该是时间所表示的维度。(维度为0,1,2)
如：input_shape=(32,10,16)考虑32个样本的一个batch，其中每个样本是10个16 维向量的序列，其中10指的是时间步
使用 TimeDistributed 来将 Dense 层独立地应用到 这 10 个时间步的每一个
这里面10指的是时间步，这就是时间分布层TimeDistributed名字的由来，他使得每一个时间步都调用一次某个相同的网络层，调用的这些网络层，都有相同的权重参数
"""
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