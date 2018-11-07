# coding=utf-8
# 输出数据曲线
# ------------
from pandas import read_csv
from matplotlib import pyplot

dataset = read_csv('data_set/air_pollution_new.csv', header=0, index_col=0)
values = dataset.values

# 需要输出的列
groups = [i for i in range(8)]
groups.remove(4)  # 删除值4，因为是字符串

i = 1
# 输出列曲线图
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i)  # 创建len(gourps)行，1列的子图，表示在第i个子图画图
    pyplot.plot(values[:, group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
pyplot.show()

