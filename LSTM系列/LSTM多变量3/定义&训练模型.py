# coding=utf-8
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


# 转成有监督数据
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # 数据序列(也将就是input)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # 预测数据（input对应的输出值）
    for i in range(0, n_out, 1):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d))' % (j + 1, i)) for j in range(n_vars)]
    # 拼接
    agg = concat(cols, axis=1)
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# 数据预处理
# --------------------------
dataset = read_csv('data_set/air_pollution_new.csv', header=0, index_col=0)
values = dataset.values

# 标签编码
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
# 保证为float
values = values.astype('float32')
# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# 转成有监督数据
reframed = series_to_supervised(scaled, 1, 1)
# 删除不预测的列
reframed.drop(reframed.columns[9:16], axis=1, inplace=True)
print(reframed.head())


# 数据准备
# --------------------------
values = reframed.values
n_train_hours = 365 * 24  # 拿一年的时间长度训练
# 划分训练数据和测试数据
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# 拆分输入输出
train_x, train_y = train[:, :-1], train[:, -1]
test_x, test_y = test[:, :-1], test[:, -1]
# reshape输入为LSTM的输入格式
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
print('train_x.shape, train_y.shape, test_x.shape, test_y.shape')
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)


# 模型定义
# -------------------------
model = Sequential()
model.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# 模型训练
# ------------------------
history = model.fit(train_x, train_y, epochs=50, batch_size=72, validation_data=(test_x, test_y), verbose=2,
                    shuffle=False)

# 输出
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# 预测
# ------------------------
yhat = model.predict(test_x)
test_x = test_x.reshape(test_x.shape[0], test_x.shape[2])
# 预测数据逆缩放
inv_yhat = concatenate((yhat, test_x[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# 真实数据逆缩放
test_y = test_y.reshape(len(test_y), 1)
inv_y = concatenate((test_y, test_x[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# 计算rmse
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE:%.3f' % rmse)
pyplot.plot(inv_yhat,label='predictions')
pyplot.plot(inv_y,label='true')
pyplot.show()
