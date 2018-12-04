from math import sin
from math import radians
from matplotlib import pyplot

# 创建一个差分序列
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

# 差分的逆转换
def inverse_difference(last_ob, value):
	return value + last_ob

# 定义一个有线性趋势的数据集
data = [sin(radians(i)) for i in range(360)] + [sin(radians(i)) for i in range(360)]
pyplot.plot(data)
pyplot.show()
# 对数据做差分
diff = difference(data, 360)
pyplot.plot(diff)
pyplot.show()
# 差分的逆转换
inverted = [inverse_difference(data[i], diff[i]) for i in range(len(diff))]
pyplot.plot(inverted)
pyplot.show()