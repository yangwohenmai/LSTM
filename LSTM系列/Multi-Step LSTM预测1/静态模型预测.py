# coding=utf-8
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot
from pandas import datetime


def parser(x):
    return datetime.strptime(x, '%Y/%m/%d')

# 把数据拆分，线性数据变成四个一组的监督型数据
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)  # 数据多了行标、列标
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    for i in range(0, n_out, 1):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# 拆分正训练+测试数据
def prepare_data(series, n_test, n_lay, n_seq):
    raw_values = series.values
    raw_values = raw_values.reshape(len(raw_values), 1)
     #转换成四个一组的监督型数据
    supervised = series_to_supervised(raw_values, n_lay, n_seq)
    supervised_values = supervised.values
    # 前3/4作为训练数据，后1/4作为预测 测试数据
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return train, test


# persistence model预测
# 用上一次观察值作为之后n_seq的预测值
# 其实只是单纯的把上一次的观测值，重复三次写入一个包含三个元素的数组，作为一个包含三个元素的预测结果
def persistence(last_ob, n_seq):
    return [last_ob for i in range(n_seq)]


# 评估persistence model
# 把由
def make_forcast(train, test, n_lay, n_seq):
    forcasts = list()
    for i in range(len(test)):
        x, y = test[i, 0:n_lag], test[i, n_lag:]
        # 这里的预测其实就是抄写上一次的观测值，把观测值变成一个数组列表
        forcast = persistence(x[-1], n_seq)
        forcasts.append(forcast)
    return forcasts


# 预测评估
# 计算预测结果的损失值，把抄写的观测值结果带入运算损失值，输出。
def evaluate_forcasts(test, forcasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = test[:, (n_lag + i)]
        predicted = [forcast[i] for forcast in forcasts]
        print('predicted')
        print(predicted)
        rmse = sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE:%f' % ((i + 1), rmse))  # 1~n_seq各个长度的预测的rmse


def plot_forcasts(series, forcasts, n_test):
    # 原始数据
    pyplot.plot(series.values)
    # 预测数据
    for i in range(len(forcasts)):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forcasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forcasts[i]
        print('xaxis')
        print(xaxis)
        print('yaxis')
        print(yaxis)
        print('series.values[off_s]')
        print(series.values[off_s])
        pyplot.plot(xaxis, yaxis, color='red')
    pyplot.show()


series = read_csv('data_set/shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# 一步数据，预测3步
n_lag = 1
n_seq = 3
n_test = 10  # 给了最后12个月，预测3个月，则能预测的次数是10，即10个3个月,即1,2,3->4 2,3,4->5 3,4,5->6 ...
train, test = prepare_data(series, n_test, n_lag, n_seq)
print('train data')
print(train)
print('test data')
print(test)
forecasts = make_forcast(train, test, n_lag, n_seq)
print('forecasts')
print(forecasts)
# 没有任何意义，只是为了教你如何进行多步的预测，数据全是根据最后观测值编造的
evaluate_forcasts(test, forecasts, n_lag, n_seq)
plot_forcasts(series, forecasts, n_test + 2)
