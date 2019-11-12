from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from pandas import datetime

# 加载数据
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# 把时间序列转换成监督学习型数据
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    print(df)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    #for i in range(n_in, 0, -1):
    #    cols.append(df.shift(i))
    #    print(cols)
    #    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    #    print(names)
    # forecast sequence (t, t+1, ... t+n)
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
def prepare_data(series, n_test, n_lag, n_seq):
    # 去掉数据的行列头
    print(series)
    print(series.values)
    raw_values = series.values
    raw_values = raw_values.reshape(len(raw_values), 1)
    print(raw_values)
    # 将基础数据转换成监督学习数据
    supervised = series_to_supervised(raw_values, n_lag, n_seq)
    print(supervised)
    # 去掉数据的行列头
    supervised_values = supervised.values
    print(supervised_values)
    # 分割训练数据和测试数据
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return train, test




# 加载数据
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
n_lag = 1
# 监督学习数据每一个sample的步长,这里每次预测3步,总步长为4
n_seq = 4
# 测试数据数量
n_test = 10
# 构建数据
train, test = prepare_data(series, n_test, n_lag, n_seq)
print(test)
print('Train: %s, Test: %s' % (train.shape, test.shape))