from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy

# 数据结构处理
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# 将数据转换为监督型学习数据，NaN值补0
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# 构建差分序列
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# 差分逆转换
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# 将数据缩放到 [-1, 1]之间的数
def scale(train, test):
    # 创建一个缩放器
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    print(train)
    # 将train从二维数组的格式转化为一个23*2的张量
    #train = train.reshape(train.shape[0], train.shape[1])
    # 使用缩放器将数据缩放到[-1, 1]之间
    train_scaled = scaler.transform(train)
    print(train_scaled)
    # transform test
    #test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# 数据逆缩放
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# 构建一个LSTM网络模型
def fit_lstm(train, batch_size, nb_epoch, neurons):
    # 将数据对中的X, y拆分开
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
    # neurons是神经元个数，batch_size是样本个数，batch_input_shape是输入形状，stateful是状态保留
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
    # 定义损失函数和优化器
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
        # shuffle=False是不混淆数据顺序
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
		model.reset_states()
	return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

# 加载数据
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# 将所有数据进行差分转换
raw_values = series.values
diff_values = difference(raw_values, 1)
print(diff_values)

# 将数据转换为监督学习型数据,此时输出的supervised_values是一个二维数组
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values
print(supervised_values)

# 将数据分割为训练集和测试集，此时分割的数据集是二维数组
train, test = supervised_values[0:-12], supervised_values[-12:]

# 将训练集和测试集都缩放到[-1, 1]之间
scaler, train_scaled, test_scaled = scale(train, test)

# # 构建一个LSTM网络模型，样本数：1，循环训练次数：3000，LSTM层神经元个数为4
lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
# 重构输入数据的形状，
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
# 使用构造的网络模型进行预测训练
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
	# 单步预测
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	yhat = forecast_lstm(lstm_model, 1, X)
	# 数据逆缩放
	yhat = invert_scale(scaler, X, yhat)
	# 差分逆转换
	yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	# store forecast
	predictions.append(yhat)
	expected = raw_values[len(train) + i + 1]
	print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

# report performance
rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.plot(raw_values[-12:])
pyplot.plot(predictions)
pyplot.show()