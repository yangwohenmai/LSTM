from pandas import Series
from sklearn.preprocessing import StandardScaler
from math import sqrt
# 手动定义一个待标准化序列
data = [1.0, 5.5, 9.0, 2.6, 8.8, 3.0, 4.1, 7.9, 6.3]
series = Series(data)
print(series)
# 准备标准化数据格式
values = series.values
values = values.reshape((len(values), 1))
# 配置标准化方法StandardScaler
# y = (x - mean) / standard_deviation
scaler = StandardScaler()
scaler = scaler.fit(values)
print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))
# 输出标准化后的数据
standardized = scaler.transform(values)

# 标准化后的数据均值为0，方差为1
scaler = scaler.fit(standardized)
print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))

print(standardized)
# 标准化数据的逆转换
inversed = scaler.inverse_transform(standardized)
print(inversed)

