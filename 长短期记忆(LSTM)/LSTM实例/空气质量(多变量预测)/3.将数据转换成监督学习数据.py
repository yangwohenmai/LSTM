from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd

pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # 获取特征值数量n_vars
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    print(df)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    # 创建8个v(t-1)作为列名
    for i in range(n_in, 0, -1):
        # 向列表cols中添加一个df.shift(1)的数据
        cols.append(df.shift(i))
        print(cols)
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        # 向列表cols中添加一个df.shift(-1)的数据
        cols.append(df.shift(-i))
        print(cols)
        if i == 0:
        	names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
        	names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    print(cols)
    # 将列表中两个张量按照列拼接起来，list(v1,v2)->[v1,v2],其中v1向下移动了一行，此时v1,v2是监督学习型数据
    agg = concat(cols, axis=1)
    print(agg)
    # 重定义列名
    agg.columns = names
    print(agg)
    # 删除空值
    if dropnan:
    	agg.dropna(inplace=True)
    return agg

# load dataset
dataset = read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
print(values)
# 对第四列“风向”进行数字编码转换
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
print(values[:,4])
# 数据转换为浮点型
values = values.astype('float32')
# 将所有数据缩放到（0，1）之间
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# 将数据格式化成监督学习型数据
reframed = series_to_supervised(scaled, 1, 1)
print(reframed.head())
# 删掉那些我们不想预测的列,axis=1列操作
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
# 最终每行数据格式如下，其中v1(t-1)~v8(t-1)表示前一天的数据，v1(t)表示当天要预测的数据：
# v1(t-1),v2(t-2),v3(t-3),v4(t-4),v5(t-5),v6(t-6),v7(t-7),v8(t-1),v1(t)
print(reframed.head())