from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot

# 加载数据
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# 把时间序列转换成监督学习型数据
def series_to_supervised(data, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    print(df)
    cols, names = list(), list()
    # 构建(t, t+1, t+2, t+3)四列监督型数据
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        print(cols)
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        print(names)
    # 将4列数据拼接在一起
    agg = concat(cols, axis=1)
    print(agg)
    agg.columns = names
    print(agg)
    # 删除NaN值
    if dropnan:
    	agg.dropna(inplace=True)
    print(agg)
    return agg

# 数据转换成监督型数据
def prepare_data(series, n_test, n_seq):
    # 去掉数据的行列头
    print(series)
    print(series.values)
    raw_values = series.values
    raw_values = raw_values.reshape(len(raw_values), 1)
    print(raw_values)
    # 将基础数据转换成监督学习数据
    supervised = series_to_supervised(raw_values, n_seq)
    print(supervised)
    # 去掉数据的行列头
    supervised_values = supervised.values
    print(supervised_values)
    # 分割训练数据和测试数据
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return train, test

# 组织预测数据
def make_forecasts(test, n_seq):
    forecasts = list()
    for i in range(len(test)):
        # X取二维数组test中每行第一列，y取每行2，3，4列
        X, y = test[i, 0:1], test[i, 1:]
        # 假装预测，取X的值3遍，作为预测值
        forecast = [X[0] for i in range(n_seq)]
        # 将假的预测值存起来
        forecasts.append(forecast)
    return forecasts

# 评估预测结果的均方差，并没有什么用
def evaluate_forecasts(test, forecasts):
    # 找三个序列评估一下预测值均方差
    for i in range(3):
        # 用test构造真实值
        actual = test[:,(i+1)]
        print(actual)
        # 用test构造预测值
        predicted = [forecast[i] for forecast in forecasts]
        print(predicted)
        rmse = sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE: %f' % ((i+1), rmse))

# 在原始数据上画出预测数据
def plot_forecasts(series, forecasts, n_test):
	# 蓝线画出真实数据
	pyplot.plot(series.values)
	# 红线画出预测值
	for i in range(len(forecasts)):
		off_s = len(series) - n_test + i - 1
		off_e = off_s + len(forecasts[i]) + 1
        # 赋值x轴坐标
		xaxis = [x for x in range(off_s, off_e)]
        # 赋值y轴坐标
		yaxis = [series.values[off_s]] + forecasts[i]
		pyplot.plot(xaxis, yaxis, color='red')
	pyplot.show()

# 加载数据
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# configure
n_lag = 1
n_seq = 3
n_test = 10
# 准备步长为4的监督型数据
train, test = prepare_data(series, n_test, 4)
# 组织预测数据
forecasts = make_forecasts(test, n_seq)
# 评估预测结果
evaluate_forecasts(test, forecasts)
# 做参照图
plot_forecasts(series, forecasts, n_test+2)