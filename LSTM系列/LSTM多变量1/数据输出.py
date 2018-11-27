# coding=utf-8
# 输出数据曲线
# ------------
from pandas import read_csv
from matplotlib import pyplot

dataset = read_csv('data_set/air_pollution_new.csv', header=0, index_col=0)
values = dataset.values

# 需要输出的列，创建一个数组，[0,1,2,3,4,5,6,7]
groups = [i for i in range(8)]
groups.remove(4)  # 删除数组中的值4，因为第四列是字符串，删除后的数组是[0,1,2,3,5,6,7]

i = 1
# 输出列曲线图
pyplot.figure()
#循环画出values的每一列数据
for group in groups:
    pyplot.subplot(len(groups), 1, i)  # 创建len(gourps)行，1列的子图，表示在第i个子图画图
    pyplot.plot(values[:, group])#画出values中第group列的数据
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
pyplot.show()

