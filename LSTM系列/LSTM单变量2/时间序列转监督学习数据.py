from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot


# load data
def parser(x):
    return datetime.strptime(x, '%Y-%m-%d')


series = read_csv('data_set/shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True,
                  date_parser=parser)

# 分成训练和测试集合
X = series.values
train, test = X[0:-12], X[-12:]

'''
步进验证模型:
其实相当于已经用train训练好了模型
之后每一次添加一个测试数据进来
1、训练模型
2、预测一次，并保存预测结构，用于之后的验证
3、加入的测试数据作为下一次迭代的训练数据
'''
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    predictions.append(history[-1])  # history[-1],就是执行预测
    history.append(test[i])  # 将新的测试数据加入模型

# 预测效果评估
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE:%.3f' % rmse)

# 画出预测+观测值
pyplot.plot(test)
pyplot.plot(predictions)
pyplot.show()
