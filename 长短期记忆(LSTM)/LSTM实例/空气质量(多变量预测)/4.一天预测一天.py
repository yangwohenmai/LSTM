from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd

pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)

"""
在最后部分，计算REM误差时有一系列奇怪的操作
猜测是因为逆缩放时对数据形状有要求，所以花力气将数据拼接成原有形状
"""
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # 获取特征值数量n_vars
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    print(df)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    # 创建8个v(t-1)作为列名
    for i in range(n_in, 0, -1):
        # 向列表cols中添加一个df.shift(1)的数据
        cols.append(df.shift(i))
        print(cols)
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        # 向列表cols中添加一个df.shift(-1)的数据
        cols.append(df.shift(-i))
        print(cols)
        if i == 0:
        	names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
        	names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    print(cols)
    # 将列表中两个张量按照列拼接起来，list(v1,v2)->[v1,v2],其中v1向下移动了一行，此时v1,v2是监督学习型数据
    agg = concat(cols, axis=1)
    print(agg)
    # 重定义列名
    agg.columns = names
    print(agg)
    # 删除空值
    if dropnan:
    	agg.dropna(inplace=True)
    return agg

# load dataset
dataset = read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
print(values)
# 对第四列“风向”进行数字编码转换
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
print(values[:,4])
# 数据转换为浮点型
values = values.astype('float32')
# 将所有数据缩放到（0，1）之间
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# 将数据格式化成监督学习型数据
reframed = series_to_supervised(scaled, 1, 1)
print(reframed.head())
# 删掉那些我们不想预测的列,axis=1列操作
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
# 最终每行数据格式如下，其中v1(t-1)~v8(t-1)表示前一天的数据，v1(t)表示当天要预测的数据：
# v1(t-1),v2(t-2),v3(t-3),v4(t-4),v5(t-5),v6(t-6),v7(t-7),v8(t-1),v1(t)
print(reframed.head())

# split into train and test sets
values = reframed.values
print(values)
# 取出一年的数据作为训练数据，剩下的做测试数据
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# 将数据分割成输入v1(t-1)~v8(t-1)和输出v1(t)，最后一列v1(t)数据作为输出数据
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# 将输入数据转换成3D张量 [samples, timesteps, features]，[n条数据，每条数据1个步长，8个特征值]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# 最终生成的数据形状，X:(8760,1,8)  Y:(8760,)
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# 设计网络结构
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# 拟合网络
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# 图像展示训练损失
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
# 使用拟合后的网络进行预测
yhat = model.predict(test_X)
print(yhat[:10,:])
print(test_X[0:6,:])
# xx = test_X[0:35000,:].reshape((17500,2 ,8))
# print(xx)

#重构预测数据形状，进行逆缩放
# 之所有下面奇怪的数据拼接操作，是因为：数据逆缩放要求输入数据的形状和缩放之前的输入保值一致
# 将3D转换为2D
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# 拼接y，x为[y,x],即将test_X中的第一列数据替换成预测出来的yhat值
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
print(inv_yhat)
# 对替换后的inv_yhat预测数据逆缩放
inv_yhat = scaler.inverse_transform(inv_yhat)
# 逆缩放后取出第一列(预测列)y
inv_yhat = inv_yhat[:,0]


# 重构真实数据形状，进行逆缩放
# 将y的形状从[35039,]转换成[35039,1]
test_y = test_y.reshape((len(test_y), 1))
# 因为test_X的第一列数据在上面被修改过，这里要重新还原一下真实数据。将test_X的第一列换成原始数据test_y值
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
# 对重构后的数据进行逆缩放
inv_y = scaler.inverse_transform(inv_y)
# 逆缩放后取出第一列(真实列)y
inv_y = inv_y[:,0]
# 计算预测列和真实列的误差RMSE值
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)