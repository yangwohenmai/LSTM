# load and plot dataset
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
# 加载数据
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# 显示开头部分行
print(series.head())
# 画图
series.plot()
pyplot.show()