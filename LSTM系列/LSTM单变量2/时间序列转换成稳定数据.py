# coding=utf-8
from pandas import read_csv
from pandas import datetime
from pandas import Series


# load data
def parser(x):
    return datetime.strptime(x, '%Y/%m/%d')


series = read_csv('data_set/shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True,
                  date_parser=parser)

# 所谓的差分，就是相邻的两个数相减，求得两个数之间的差，，一两个数之间的差作为一个数组，这样的数组体现了相邻两个数的变化情况
# 所谓的去趋势也就是一个说法而已，可以忽略
# 做差分，去趋势，获得差分序列
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]  # 当前时间步t的值减去时间步t-interval的值
        diff.append(value)
    return Series(diff)#Series这个方法是把一个数组建立起一个一一对应的索引，参考：https://blog.csdn.net/brucewong0516/article/details/79196902


# 这就是反过来算一遍，拿治最后一个数减去差分的数，就还原回原来的数组了，总的来说这个例子没什么卵用
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
