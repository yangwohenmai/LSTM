# Stacked LSTM for international airline passengers problem with memory
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
"""
堆叠LSTM时，需要将上层的输入形状向下继续传递，设置return_sequences=True参数
差异：
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
"""
# 将数据截取成3个一组的监督学习格式
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# 定义随机种子，以便重现结果
numpy.random.seed(7)
# 加载数据
dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
# 缩放数据
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# 分割2/3数据作为测试
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# 预测数据步长为3,三个预测一个，3->1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# 重构输入数据格式 [samples, time steps, features] = [93,3,1]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# 构建 LSTM 网络
batch_size = 1
model = Sequential()
# 堆叠两层网络，参数return_sequences=True表示将上层的输入形状向下继续传递
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# 网络训练一个周期，循环训练100次
for i in range(100):
	model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
    # 每次训练完都重置网络状态
	model.reset_states()
# 对训练数据的Y进行预测
trainPredict = model.predict(trainX, batch_size=batch_size)
# 重置网络状态
model.reset_states()
# 对测试数据的Y进行预测
testPredict = model.predict(testX, batch_size=batch_size)
# 对数据进行逆缩放
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# 计算RMSE误差
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# 构造一个和dataset格式相同的数组，共145行，dataset为总数据集，把预测的93行训练数据存进去
trainPredictPlot = numpy.empty_like(dataset)
# 用nan填充数组
trainPredictPlot[:, :] = numpy.nan
# 将训练集预测的Y添加进数组，从第3位到第93+3位，共93行
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# 构造一个和dataset格式相同的数组，共145行，把预测的后44行测试数据数据放进去
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
# 将测试集预测的Y添加进数组，从第94+4位到最后，共44行
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# 画图
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()