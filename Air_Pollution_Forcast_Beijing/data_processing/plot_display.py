#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
Time:17/1/1
---------------------------
Question: 读取处理过的文件数据并绘图
---------------------------
"""
import pandas as pd
import matplotlib.pyplot as plt
from Air_Pollution_Forcast_Beijing.util import PROCESS_LEVEL1

pd.options.display.expand_frame_repr = False

dataset = pd.read_csv(PROCESS_LEVEL1, header=0, index_col=0)

# loads the 'pollution.csv' and treat each column as a separate subplot
values = dataset.values
groups = [0, 1, 2, 3, 5, 6, 7]

i = 1
plt.figure()
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])
    plt.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
plt.show()

# print(dataset.head())
