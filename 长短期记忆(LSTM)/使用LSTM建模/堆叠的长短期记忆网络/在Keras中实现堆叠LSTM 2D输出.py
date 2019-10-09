# Example of one output for whole sequence
from keras.models import Sequential
from keras.layers import LSTM
from numpy import array
"""
网络中不加return_sequences=True参数时，一个(1,3,1)的3D输入，进入神经网络后，变成一个(1,1)的2D输出
此时如果用for循环对网络进行迭代计算，则当前输出数据形状，不能再作为下次计算的输入数据
"""
# define model where LSTM is also output layer
model = Sequential()
model.add(LSTM(1, input_shape=(3,1)))
model.compile(optimizer='adam', loss='mse')
# input time steps
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
# make and show prediction
print(model.predict(data))
print(data.shape)
print(model.predict(data).shape)