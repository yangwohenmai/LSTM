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
from numpy import array

# 修正数据格式
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# 将时间序列转换成监督学习数据
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# 数据差分法
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# 数据特征工程,差分，缩放，分割
def prepare_data(series, n_test, n_lag, n_seq):
    # 提取文本中的数据
    raw_values = series.values
    # 对数据进行差分计算
    diff_series = difference(raw_values, 1)
    # 提取差分后的数据
    diff_values = diff_series.values
    print(diff_values)
    # 重构成n行一列的数据
    diff_values = diff_values.reshape(len(diff_values), 1)
    print(diff_values)
    # 定义数据缩放在(-1，1)之间
    scaler = MinMaxScaler(feature_range=(-1, 1))
    print(scaler)
    # 对数据进行缩放
    scaled_values = scaler.fit_transform(diff_values)
    print(scaled_values)
    # 将缩放后的数据重构成n行一列的数据
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    print(scaled_values)
    # 将数据构建成步长为n_seq的监督学习型数据
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    print(supervised)
    supervised_values = supervised.values
    # 将数据分割出n_test条作为测试数据
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test

# 你和一个LSTM网络，训练数据
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # 每个4位序列中，第1位作为x，后3位作为预测值y
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    # 重构训练数据结构->[samples, timesteps, features]->[22,1,1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    print(X)
    print(y)
    # 网络结构
    model = Sequential()
    # 一个神经元， batch_input_shape(1,1,1)，传递序列状态
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # 开始训练
    for i in range(nb_epoch):
        # 数据训练1次，每次训练1组数据，不混淆序列顺序
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        # 每次训练完初始化网络状态（不是权重）
        model.reset_states()
    return model

# LSTM 单步预测
def forecast_lstm(model, X, n_batch):
	# 重构输入形状 (1,1,1) [samples, timesteps, features]
	X = X.reshape(1, 1, len(X))
	# 预测张量形状为 (1,3)
	forecast = model.predict(X, batch_size=n_batch)
	# 将预测结果[[XX,XX,XX]]转换成list数组
	return [x for x in forecast[0, :]]

# 用模型进行预测
def make_forecasts(model, n_batch, test, n_lag, n_seq):
    forecasts = list()
    # 对X值进行逐个预测
    for i in range(len(test)):
        # X, y = test[i, 0:n_lag], test[i, n_lag:]
        X = test[i, 0:n_lag]
        # LSTM 单步预测
        forecast = forecast_lstm(model, X, n_batch)
        # 存储预测数据
        forecasts.append(forecast)
    return forecasts

# 对预测数据逆差分
def inverse_difference(last_ob, forecast):
	# invert first forecast
	inverted = list()
	inverted.append(forecast[0] + last_ob)
	# propagate difference forecast using inverted first value
	for i in range(1, len(forecast)):
		inverted.append(forecast[i] + inverted[i-1])
	return inverted

# 对预测后的数据逆转换
def inverse_transform(series, forecasts, scaler, n_test):
	inverted = list()
	for i in range(len(forecasts)):
		# create array from forecast
		forecast = array(forecasts[i])
		forecast = forecast.reshape(1, len(forecast))
		# 数据逆缩放
		inv_scale = scaler.inverse_transform(forecast)
		inv_scale = inv_scale[0, :]
		# 数据逆差分
		index = len(series) - n_test + i - 1
		last_ob = series.values[index]
		inv_diff = inverse_difference(last_ob, inv_scale)
		# 存储转换后的数据
		inverted.append(inv_diff)
	return inverted

# 评估预测结果的均方差
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = [row[i] for row in test]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))

# 作图
def plot_forecasts(series, forecasts, n_test):
	# plot the entire dataset in blue
	pyplot.plot(series.values)
	# plot the forecasts in red
	for i in range(len(forecasts)):
		off_s = len(series) - n_test + i - 1
		off_e = off_s + len(forecasts[i]) + 1
		xaxis = [x for x in range(off_s, off_e)]
		yaxis = [series.values[off_s]] + forecasts[i]
		pyplot.plot(xaxis, yaxis, color='red')
	# show the plot
	pyplot.show()

# 加载数据
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# 参数配置
n_lag = 1       # 用一个数据
n_seq = 3       # 预测三个数据
n_test = 10     # 测试数据为10组
n_epochs = 1500 # 训练1500次
n_batch = 1     # 每次训练几组数据
n_neurons = 1   # 神经节点为1
# 数据差分，缩放，重构成监督学习型数据
scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
# 拟合模型
model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
# 开始预测
forecasts = make_forecasts(model, n_batch, test, n_lag, n_seq)
# 将预测后的数据逆转换
forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
# 从测试数据中分离出y对应的真实值
actual = [row[n_lag:] for row in test]
# 对真实值逆转换
actual = inverse_transform(series, actual, scaler, n_test+2)
# 评估预测值和真实值的RSM
evaluate_forecasts(actual, forecasts, n_lag, n_seq)
# 作图
plot_forecasts(series, forecasts, n_test+2)