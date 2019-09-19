from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
"""
https://machinelearningmastery.com/5-step-life-cycle-long-short-term-memory-models-keras/
"""
# 创建一个0.1~0.9的序列[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
length = 10
sequence = [i/float(length) for i in range(length)]
print(sequence)
# 构建一个X->y的映射关系
"""DataFrame方法可以把一个数组序列转换成一个有序编码的矩阵序列
     0
0  0.0
1  0.1
2  0.2
3  0.3
4  0.4
5  0.5
6  0.6
7  0.7
8  0.8
9  0.9
"""
df = DataFrame(sequence)
print(df)
"""concat是一个链接方法，把多个数据拼接起来，axis=1就相当于把数据按照行对应拼接起来
DataFrame.shift是一个位移函数，df.shift(1)就相当于将df序列向下整体移动一位，第一位用NAN值补上。
     0    0
0  NaN  0.0
1  0.0  0.1
2  0.1  0.2
3  0.2  0.3
4  0.3  0.4
5  0.4  0.5
6  0.5  0.6
7  0.6  0.7
8  0.7  0.8
9  0.8  0.9
"""
df = concat([df.shift(1), df], axis=1)
print(df)
"""删除数据中为NAN的数据
     0    0
1  0.0  0.1
2  0.1  0.2
3  0.2  0.3
4  0.3  0.4
5  0.4  0.5
6  0.5  0.6
7  0.6  0.7
8  0.7  0.8
9  0.8  0.9
"""
df.dropna(inplace=True)
print(df)
# 使用reshape方法，把序列转换为LSTM可识别的数组格式
"""将df这个矩阵序列中的有用值提取出来，变成一个二维的数组数据
[[0.  0.1]
 [0.1 0.2]
 [0.2 0.3]
 [0.3 0.4]
 [0.4 0.5]
 [0.5 0.6]
 [0.6 0.7]
 [0.7 0.8]
 [0.8 0.9]]
"""
values = df.values
print(values)
"""values[:, 0]是将数组中第0列的所有行提取，赋值给X，values[:, 1]则是获取第1列
[0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8]
(9,)
"""
X, y = values[:, 0], values[:, 1]
print(X)
print(X.shape)
"""pandas.reshape将一个一维数组转换成一个三维的数据类型，神经网络的输入数据都是三维的
[[[0. ]]
 [[0.1]]
 [[0.2]]
 [[0.3]]
 [[0.4]]
 [[0.5]]
 [[0.6]]
 [[0.7]]
 [[0.8]]]
 (9, 1, 1)
 (9, 1, 1)代表9个样本，每个样本步长为1，并且有1个特征值
 """
X = X.reshape(len(X), 1, 1)
print(X.shape)
print(X)

# 1. 定义网络类型，Sequential是一个参数容器
model = Sequential()
# 由存储器单元组成的LSTM循环层称为LSTM（），input_shape（步长，特征值），可以指定input_shape参数，该参数需要包含时间步长和特征值的元组
model.add(LSTM(10, input_shape=(1,1)))
# 通常跟随LSTM层并用于输出预测的完全连接层称为Dense（）。
model.add(Dense(1))
# 2. 编译网络，设置损失参数,优化器是Adam，损失函数是均方差
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