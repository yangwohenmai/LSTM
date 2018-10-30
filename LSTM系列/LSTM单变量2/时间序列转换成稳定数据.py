# coding=utf-8
from pandas import read_csv
from pandas import datetime
from pandas import Series


# load data
def parser(x):
    return datetime.strptime(x, '%Y-%m-%d')


series = read_csv('data_set/shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True,
                  date_parser=parser)


# 做差分，去趋势，获得差分序列
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]  # 当前时间步t的值减去时间步t-interval的值
        diff.append(value)
    return Series(diff)


# 将预测值进行逆处理，得到真实的销售预测
def inverse_difference(history, yhat, interval=1):  # 历史数据，预测数据，差分间隔
    return yhat + history[-interval]


# 数据处理

# 将数据转换成稳定的
differenced = difference(series, 1)
print(differenced.head())

# 逆处理，从差分逆转得到真实值
inverted = list()
for i in range(len(differenced)):
    value = inverse_difference(series, differenced[i], len(series) - i)
    inverted.append(value)
inverted = Series(inverted)
print(inverted.head())
