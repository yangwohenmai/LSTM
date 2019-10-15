from pandas import read_csv
from pandas import datetime
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
"""
通过配置MinMaxScaler的feature_range参数，可以将数据缩放在任意范围内
"""

# 数据格式处理
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
print(series)

# transform scale
X = series.values
X = X.reshape(len(X), 1)
# feature_range定义数据缩放范围
scaler = MinMaxScaler(feature_range=(-1, 1))
# 对数据进行适配，找到最大最小值等特征，便于后续转换
scaler = scaler.fit(X)
# 开始转换数据,输出一个二维数组
scaled_X = scaler.transform(X)
print(scaled_X)
# 将二维数组转换成序列
scaled_series = Series(scaled_X[:, 0])
print(scaled_series)

# 将缩放后的数据反向转换成原值
inverted_X = scaler.inverse_transform(scaled_X)
inverted_series = Series(inverted_X[:, 0])
print(inverted_series)