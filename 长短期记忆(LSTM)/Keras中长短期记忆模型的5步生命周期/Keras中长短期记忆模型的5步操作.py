from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
"""
https://machinelearningmastery.com/5-step-life-cycle-long-short-term-memory-models-keras/
"""
# 创建一个0.1~0.9的序列
length = 10
sequence = [i/float(length) for i in range(length)]
print(sequence)
# 以下部分开始构建一个X->y的映射关系
#将序列转换为竖直排列的格式
df = DataFrame(sequence)
print(df)
# 创建一个监督序列数据，axis=1表示对列操作(拼接成多列)，axis=0表示对行操作(拼接成一列多行)
# concat将多个df序列合并成一个集合，shift(1)将当前列向下移动，shift(-1)向上移动
df = concat([df.shift(1), df], axis=1)
print(df)
# 删除有na值的行
df.dropna(inplace=True)
print(df)
# 使用reshape方法，把监督型序列转换为LSTM可识别的数组格式
values = df.values
print(values)
X, y = values[:, 0], values[:, 1]
print(X)
print(X.shape)
X = X.reshape(len(X), 1, 1)
print(X.shape)
print(X)
# 1. 定义网络类型
model = Sequential()
model.add(LSTM(10, input_shape=(1,1)))
model.add(Dense(1))
# 2. 编译网络，设置损失参数
model.compile(optimizer='adam', loss='mean_squared_error')
# 3. 调用网络开始训练模型，对数据训练1000次
history = model.fit(X, y, epochs=1000, batch_size=len(X), verbose=0)
# 4. 评估网络
loss = model.evaluate(X, y, verbose=0)
print(loss)
# 5. 利用训练好的模型，带入原始的X进行单步预测
predictions = model.predict(X, verbose=0)
print(predictions[:, 0])


# 构建一个新序列，对预测模型进行二次预测
# 创建一个0.1~0.9的序列
length = 10
sequence = [(i+5)/float(length) for i in range(length)]
print(sequence)
# 构建一个X->y的映射关系
df = DataFrame(sequence)
df = concat([df.shift(1), df], axis=1)
df.dropna(inplace=True)
print(df)
# 使用reshape方法，把序列转换为LSTM可识别的数组格式
values = df.values
X, y = values[:, 0], values[:, 1]
X = X.reshape(len(X), 1, 1)
predictions = model.predict(X, verbose=0)
print(predictions[:, 0])
#原始序列[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
#预测数列[0.5880783  0.68985283 0.794536   0.9015892  1.0105045  1.1208173 1.2321128  1.3440294  1.4562595 ]
#正确序列[0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]