# Example of one output for each input time step
from keras.models import Sequential
from keras.layers import LSTM
from numpy import array
"""
想要堆叠，只需在网络中设置return_sequences=True参数即可
设置参数后，当前网络的输出会保持和输入数据形状一样，以便后续网络继续处理数据
输入的数据形状是(1,3,1)，输出的数据形状也是(1,3,1)
"""
# define model where LSTM is also output layer
model = Sequential()
model.add(LSTM(1, return_sequences=True, input_shape=(3,1)))
model.compile(optimizer='adam', loss='mse')
# input time steps
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
# make and show prediction
print(model.predict(data))
print(data.shape)
print(model.predict(data).shape)