from pandas import Series
from sklearn.preprocessing import MinMaxScaler
# 手动定义一串序列
data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
series = Series(data)
print(series)
# 准备归一化数据格式
values = series.values
values = values.reshape((len(values), 1))
# 配置数据归一化方法MinMaxScaler
# y = (x - min) / (max - min)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)
print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
# 输出归一化数据
normalized = scaler.transform(values)
print(normalized)
# 归一化逆转换
inversed = scaler.inverse_transform(normalized)
print(inversed)