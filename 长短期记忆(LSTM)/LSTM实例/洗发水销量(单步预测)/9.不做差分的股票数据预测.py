from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
"""
单步预测是指一次预测一条数据
训练时将训练数据全部输入，
预测时，获取训练好的模型，每次输入一个数据X，预测出对应的
"""
# 数据结构处理
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# 将数据转换为监督型学习数据，NaN值补0
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# 构建差分序列
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i]
		diff.append(value)
	return Series(diff)

# 差分逆转换
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# 将数据缩放到 [-1, 1]之间的数
def scale(train, test):
    # 创建一个缩放器
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    print(train)
    # 将train从二维数组的格式转化为一个23*2的张量
    #train = train.reshape(train.shape[0], train.shape[1])
    # 使用缩放器将数据缩放到[-1, 1]之间
    train_scaled = scaler.transform(train)
    print(train_scaled)
    # transform test
    #test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# 数据逆缩放，scaler是之前生成的缩放器，X是一维数组，y是数值
def invert_scale(scaler, X, y):
	# 将x，y转成一个list列表[x,y]->[0.26733207, -0.025524002]
	# [y]可以将一个数值转化成一个单元素列表
	new_row = [x for x in X] + [y]
	#new_row = [X[0]]+[y]
	# 将列表转化为一个,包含两个元素的一维数组，形状为(2,)->[0.26733207 -0.025524002]
	array = numpy.array(new_row)
	print(array.shape)
	# 将一维数组重构成形状为(1,2)的，1行、每行2个元素的，2维数组->[[ 0.26733207 -0.025524002]]
	array = array.reshape(1, len(array))
	# 逆缩放输入的形状为(1,2),输出形状为(1,2) -> [[ 73 15]]
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# 构建一个LSTM网络模型，并训练
def fit_lstm(train, batch_size, nb_epoch, neurons):
    # 将数据对中的X, y拆分开，形状为[23*1]
    X, y = train[:, 0:-1], train[:, -1]
    # 将2D数据拼接成3D数据，形状为[23*1*1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    # neurons是神经元个数，batch_size是样本个数，batch_input_shape是输入形状，
    # stateful是状态保留
    # 1.同一批数据反复训练很多次，可保留每次训练状态供下次使用
    # 2.不同批数据之间有顺序关联，可保留每次训练状态
    # 3.不同批次数据，数据之间没有关联
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    # 定义损失函数和优化器
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        # shuffle=False是不混淆数据顺序
        model.fit(X, y, epochs=10, batch_size=batch_size, verbose=1, shuffle=False)
        # 每训练完一个轮回，重置一次网络
        model.reset_states()
    return model

# 开始单步预测，model是训练好的模型，batch_size是时间步，X是一个一维数组
def forecast_lstm(model, batch_size, X):
	# 将形状为(1,)的，包含一个元素的一维数组X，构造成形状为(1,1,1)的3D张量
	X = X.reshape(1, 1, len(X))
	# 输出yhat形状为(1,1)的二维数组
	yhat = model.predict(X, batch_size=batch_size)
	# 返回二维数组中，第一行一列的yhat的数值
	return yhat[0,0]




# 加载数据
series = read_csv('stocktest.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# 最后N条数据作为测试数据
testNum = 20

# 将所有数据进行差分转换
raw_values = series.values
diff_values = difference(raw_values, 1)

# 将数据转换为监督学习型数据,此时输出的supervised_values是一个二维数组
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# 将数据分割为训练集和测试集，此时分割的数据集是二维数组
train, test = supervised_values[0:-testNum], supervised_values[-testNum:]

# 将训练集和测试集都缩放到[-1, 1]之间
scaler, train_scaled, test_scaled = scale(train, test)


# 构建一个LSTM网络模型，并训练，样本数：1，循环训练次数：3000，LSTM层神经元个数为4
lstm_model = fit_lstm(train_scaled, 1, 10, 8)
# 重构输入数据的形状，
print(train_scaled)
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
print(train_reshaped)

# 使用构造的网络模型进行预测训练
lstm_model.predict(train_reshaped, batch_size=1)
#print(lstm_model.predict(train_reshaped, batch_size=1))
# 遍历测试数据，对数据进行单步预测
predictions = list()
for i in range(len(test_scaled)):
	# 将(12,2)的2D训练集test_scaled拆分成X,y；
	# 其中X是第i行的0到-1列，形状是(1,)的包含一个元素的一维数组；y是第i行，倒数第1列，是一个数值；
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	# 将训练好的模型lstm_model，X变量，传入预测函数，定义步长为1，
	yhat = forecast_lstm(lstm_model, 1, X)
	print(yhat.shape)
	# 对预测出的y值逆缩放
	yhat = invert_scale(scaler, X, yhat)
	# 对预测出的y值逆差分转换
	#yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	# 存储预测的y值
	predictions.append(yhat)
	# 获取真实的y值
	expected = raw_values[len(train) + i + 1]
    #
	print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

# 求真实值和预测值之间的标准差
rmse = sqrt(mean_squared_error(raw_values[-testNum:], predictions))
print('Test RMSE: %.3f' % rmse)
# 作图展示
pyplot.plot(raw_values[-testNum:])
pyplot.plot(predictions)
pyplot.show()


