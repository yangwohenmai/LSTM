# coding=utf-8

from pandas import read_csv
from pandas import datetime


def parser(x):
    return datetime.strptime(x, '%Y %m %d %H')

#加载数据
dataset = read_csv('data_set/air_pollution.csv', parse_dates=[['year', 'month', 'day', 'hour']], index_col=0,
                   date_parser=parser)
# axis=1,表示删除列；inplace=True,直接在原DataFrame上执行删除
dataset.drop('No', axis=1, inplace=True)

# 手动设置每一列的label
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
# 将NA替换为0
dataset['pollution'].fillna(0, inplace=True)
# 删除最开始的24条数据
dataset = dataset[24:]
print (dataset.head())

# 保存处理后数据
dataset.to_csv('data_set/air_pollution_new.csv')
