#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
Time:17/1/1
---------------------------
Question: Series  --->  Supervised Learning Problem
    1、时间序列预测，以前一时刻（t-1）的所有数据预测当前时刻（t）的值

    X = PM2.5(t-1)  pollution(t-1) ,dew(t-1) ,temp(t-1) ,press(t-1) ,wnd_dir(t-1) ,wnd_spd(t-1) ,snow(t-1) ,rain(t-1)
    Y = PM2.5(t)

    2、在做inversed_transformed 时，需要注意的是所有的维度需要保持一致
---------------------------
"""
import pandas as pd
from Air_Pollution_Forcast_Beijing.util import PROCESS_LEVEL1
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from Air_Pollution_Forcast_Beijing.model.series_to_supervised_learning import series_to_supervised
pd.options.display.expand_frame_repr = False


dataset = pd.read_csv(PROCESS_LEVEL1, header=0, index_col=0)
dataset_columns = dataset.columns
values = dataset.values
# print(dataset)

# 对第四列（风向）数据进行编码，也可进行 哑编码处理
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
values = values.astype('float32')

# 对数据进行归一化处理, valeus.shape=(, 8),inversed_transform时也需要8列
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# 将序列数据转化为监督学习数据
reframed = series_to_supervised(scaled, dataset_columns, 1, 1)
# 只考虑当前时刻(t)的前一时刻（t-1）的PM2.5值
reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)

values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# 监督学习结果划分,test_x.shape = (, 8)
train_x, train_y = train[:, :-1], train[:, -1]
test_x, test_y = test[:, :-1], test[:, -1]

# 为了在LSTM中应用该数据，需要将其格式转化为3D format，即[Samples, timesteps, features]
train_X = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_X = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
