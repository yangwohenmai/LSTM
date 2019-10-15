from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat

# 构造监督学习数据结构
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    print(df)
    # 该行代码生成一个数组，将df向下移动一位作为第一个元素
    # 用这种特殊的代码写法是为了生成一个数组列表的数据结构，而不是一个序列
    # columns = df.shift(1)生成的是一个序列，而不是列表
    columns = [df.shift(i) for i in range(1, lag+1)]
    print(columns)
    print(df)
    # df作为第二个元素
    columns.append(df)
    # 将df.shift(1)，df按列合并，构成监督型数据
    df = concat(columns, axis=1)
    print(df)
    # 将NaN位置补零
    df.fillna(0, inplace=True)
    return df


# 数据格式处理
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')


series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# transform to supervised learning
X = series.values
print(X)
supervised = timeseries_to_supervised(X, 1)
print(supervised)