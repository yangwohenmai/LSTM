from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# 创建一个0.1~0.9的序列
length = 10
sequence = [i/float(length) for i in range(length)]
print(sequence)
# 构建一个X->y的映射关系
df = DataFrame(sequence)
df = concat([df.shift(1), df], axis=1)
df.dropna(inplace=True)
# 使用reshape方法，把序列转换为LSTM可识别的数组格式
values = df.values
X, y = values[:, 0], values[:, 1]
X = X.reshape(len(X), 1, 1)
# 1. 定义网络类型
model = Sequential()
model.add(LSTM(10, input_shape=(1,1)))
model.add(Dense(1))
# 2. 编译网络，设置损失参数
model.compile(optimizer='adam', loss='mean_squared_error')
# 3. 调用网络开始训练模型
history = model.fit(X, y, epochs=1000, batch_size=len(X), verbose=0)
# 4. 评估网络
loss = model.evaluate(X, y, verbose=0)
print(loss)
# 5. 利用训练好的模型，带入原始的X进行单步预测
predictions = model.predict(X, verbose=0)
print(predictions[:, 0])



# 创建一个0.1~0.9的序列
length = 10
sequence = [(i+5)/float(length) for i in range(length)]
print(sequence)
# 构建一个X->y的映射关系
df = DataFrame(sequence)
df = concat([df.shift(1), df], axis=1)
df.dropna(inplace=True)
# 使用reshape方法，把序列转换为LSTM可识别的数组格式
values = df.values
X, y = values[:, 0], values[:, 1]
X = X.reshape(len(X), 1, 1)
predictions = model.predict(X, verbose=0)
print(predictions[:, 0])