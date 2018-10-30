# coding=utf-8
from pandas import read_csv
from pandas import datetime
from pandas import Series
from sklearn.preprocessing import MinMaxScaler


# load data
def parser(x):
    return datetime.strptime(x, '%Y-%m-%d')


series = read_csv('data_set/shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True,
                  date_parser=parser)
print (series.head())

# 缩放
X = series.values
X = X.reshape(len(X), 1)  # MinMaxScaler函数需要矩阵作为输入，所以reshape数据为矩阵
scaler = MinMaxScaler(feature_range=(-1, 1))  # 定义缩放范围
scaler = scaler.fit(X)  # 缩放数据
scalered_X = scaler.transform(X)
scalered_series = Series(scalered_X[:, 0])
print (scalered_series.head())

# 逆缩放
inverted_X = scaler.inverse_transform(scalered_X)
inverted_series = Series(inverted_X[:, 0])
print (inverted_series.head())
