# coding=utf-8
from pandas import read_csv
from pandas import datetime
from pandas import Series
from sklearn.preprocessing import MinMaxScaler


# load data
def parser(x):
    return datetime.strptime(x, '%Y/%m/%d')


series = read_csv('data_set/shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True,
                  date_parser=parser)
print (series.head())

# 所谓缩放，就是把一组数组中的数字都变成[-1,1]范围的数字，取数组中最大的那个数组，令其为1,最小的数字，令其为-1，
# 剩下的数字根据比例关系，在[-1,1]中给其找一个对应值
# 缩放
X = series.values
X = X.reshape(len(X), 1)  # MinMaxScaler函数需要矩阵作为输入，所以reshape数据为矩阵，因为是一维数组，所以生成的是n行1列的一个矩阵
scaler = MinMaxScaler(feature_range=(-1, 1))  # 定义缩放范围，-1,1是数据缩放的范围
scaler = scaler.fit(X)  # 调用缩放数据的fun
scalered_X = scaler.transform(X)#转换成一个[-1,1]区间的矩阵
scalered_series = Series(scalered_X[:, 0])#把矩阵序列化成列表
print (scalered_series.head())

# 逆缩放，反着来一遍，转换回去
inverted_X = scaler.inverse_transform(scalered_X)#把数值为[-1,1]之间的矩阵转换成正常数据的矩阵
inverted_series = Series(inverted_X[:, 0])#把矩阵转换成列表
print (inverted_series.head())
