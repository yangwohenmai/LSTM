# coding=utf-8
from pandas import read_csv
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from pandas import concat
from sklearn.preprocessing import LabelEncoder


# 转换成有监督数据
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):  # n_in,n_out相当于lag
    n_vars = 1 if type(data) is list else data.shape[1]  # 变量个数
    df = DataFrame(data)
    print('待转换数据')
    print(df.head())
    cols, names = list(), list()
    # 输入序列(t-n. ... , t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        print('shift数据')
        print(cols[0][0:5])
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        print('names数据')
        print(names[0:5])
        # 预测序列(t, t+1, ... , t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:  # t时刻
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # 拼接
    agg = concat(cols, axis=1)
    print('拼接')
    print(agg[0:5])
    agg.columns = names
    # 将空值NaN行删除
    if dropnan:
        agg.dropna(inplace=True)
    return agg


dataset = read_csv('data_set/air_pollution_new.csv', header=0, index_col=0)
values = dataset.values
print('原始数据')
print(values[0:5])


# 由于4列的风向是标签，编码成整数
encoder = LabelEncoder()  # 简单来说 LabelEncoder 是对不连续的数字或者文本进行编号
values[:, 4] = encoder.fit_transform(values[:, 4])
print('标签编码')
print(values[0:5])

# 使所有数据是float类型
values = values.astype('float32')
# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
print('缩放')
print(scaled[0:5])

# 变成有监督
reframed = series_to_supervised(scaled, 1, 1)
print('有监督')
print(reframed[0:5])

# 删除不预测的列
reframed.drop(reframed.columns[9:16], axis=1, inplace=True)
print('删除不预测的列')
print(reframed.head())
