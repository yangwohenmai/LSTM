from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
"""
在没有时间分布层TimeDistributed封装器的时候，如果想要输出一个形状为(1, 5, 1)：
[[[0. ]
  [0.2]
  [0.4]
  [0.6]
  [0.8]]]
的向量时，需要将这个向量“展平”成形状为(1, 5)的向量：[[0.  0.2 0.4 0.6 0.8]]，然后用Dense(5)来输出5个连续的值。这里(1, 5)其实忽略了“5个步长，每个步长一个特征值，即(1, 5, 1)”，这个特性，而单单输出了“一个序列，序列中包含5个值”

在使用TimeDistributed封装器的时候，就不需要进行这种“展平”操作，直接使用TimeDistributed(Dense(1))，就可以输出一个(1, 5, 1)的数据。

使用TimeDistributed来将Dense层独立地应用到这5个时间步的每一个，这就是时间分布层TimeDistributed名字的由来，他使得每一个时间步都调用一次某个相同的网络层，调用的这些网络层，都有相同的权重参数
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