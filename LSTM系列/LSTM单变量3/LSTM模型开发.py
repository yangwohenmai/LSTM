
"""长短期记忆网络（LSTM）是一种回归神经网络（RNN）。

这种类型的网络的一个好处是它可以学习和记住长序列，并且不依赖于预先指定的窗口滞后观察作为输入。

在Keras中，这被称为有状态，并且涉及在定义LSTM层时将“ 有状态 ”参数设置为“ True ”。

默认情况下，Keras中的LSTM层维护一个批次内的数据之间的状态。一批数据是来自训练数据集的固定大小的行数，
用于定义在更新网络权重之前要处理的模式数。默认情况下，批次之间的LSTM层中的状态被清除，
因此我们必须使LSTM成为有状态。通过调用reset_states（）函数，这使我们能够对LSTM层的状态何时被清除进行细粒度控制。

LSTM层期望输入在具有以下维度的矩阵中：[ 样本，时间步长，特征 ]。

样本：这些是来自域的独立观察，通常是数据行。
时间步长：这些是给定观察的给定变量的单独时间步长。
特征：这些是在观察时观察到的单独测量。
我们在如何为网络构建Shampoo Sales数据集方面具有一定的灵活性。我们将保持简单并构建问题，
因为原始序列中的每个步骤都是一个单独的样本，具有一个时间步长和一个特征。

鉴于训练数据集被定义为X输入和y输出，它必须重新整形为Samples / TimeSteps / Features格式，例如："""


#X, y = train[:, 0:-1], train[:, -1]
#X = X.reshape(X.shape[0], 1, X.shape[1])



"""必须在LSTM层中使用“ batch_input_shape ”参数指定输入数据的形状作为元组，该元组指定读取每个批次的预期观察数，时间步数和特征数。

批量通常远小于样品总数。它与时期的数量一起定义了网络学习数据的速度（权重更新的频率）。

定义LSTM层的最后一个导入参数是神经元的数量，也称为内存单元或块的数量。这是一个相当简单的问题，1到5之间的数字就足够了。

下面的行创建了一个LSTM隐藏图层，该图层还通过“ batch_input_shape ”参数指定输入图层的期望值。"""


#layer = LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True)

#neurons=神经元个数，即记忆单元个数，1~5 就很高效了
#batch_input_shape=表示每个batch需要读取的数据格式（batch_size=每batch读的数据行数，步长，属性数）
#batch通常和epoch一起
#epochs：确定网络学习数据快慢，即权重的更新频率


"""网络需要输出层中的单个神经元具有线性激活，以预测下一时间步骤的洗发水销售数量。

一旦指定了网络，就必须使用后端数学库（例如TensorFlow或Theano）将其编译为有效的符号表示。

在编译网络时，我们必须指定一个损失函数和优化算法。我们将使用“ mean_squared_error ”作为损失函数，
因为它与我们感兴趣的RMSE紧密匹配，以及有效的ADAM优化算法。

使用Sequential Keras API定义网络，下面的代码片段创建并编译网络。"""


#model = Sequential()#定义网络模型
#model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))#添加一个LSTM隐藏图层
#model.add(Dense(1))#再用 model.add 添加神经层，添加的是 Dense 全连接神经层，所谓全连接就是每个节点都和上一层的节点有连接
#model.compile(loss='mean_squared_error', optimizer='adam')#mean_squared_error ”作为损失函数，adam是优化算法

"""编译后，它可以适合训练数据。由于网络是有状态的，我们必须控制何时重置内部状态。
因此，我们必须在所需数量的时期内一次手动管理一个时期的训练过程。

默认情况下，一个纪元内的样本在暴露给网络之前进行混洗。同样，这对于LSTM来说是不合需要的，因为我们希望网络在通过观察序列学习时建立状态。
我们可以通过将“ shuffle ” 设置为“ False ” 来禁用样本的混洗。

此外，默认情况下，网络会在每个时代结束时报告有关模型学习进度和技能的大量调试信息。我们可以通过将“ verbose ”参数设置为“ 0 ” 级别来禁用它。

然后我们可以在训练时期结束时重置内部状态，为下一次训练迭代做好准备。

下面是一个手动使网络适应训练数据的循环。"""


#for i in range(nb_epoch):
#	model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
#	model.reset_states()
#verbose设置为0可以禁用网络在每个时代结束时报告有关模型学习进度和技能的大量调试信息
#默认情况下，一个周期内的样本在暴露给网络之前进行混洗，可以通过设置shuffle = False来禁用样本混洗


"""综上所述，我们可以定义一个名为fit_lstm（）的函数来训练和返回LSTM模型。作为参数，它将训练数据集置于监督学习格式，批量大小，
多个时期和许多神经元中。"""


def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]#获取数据集
	X = X.reshape(X.shape[0], 1, X.shape[1])#转换为矩阵
#	model = Sequential()#调用模型
#	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))#添加模型，设置模型参数
#	model.add(Dense(1))#再用 model.add 添加神经层，添加的是 Dense 全连接神经层
#	model.compile(loss='mean_squared_error', optimizer='adam')#定义损失函数和优化算法
#	for i in range(nb_epoch):#按照给定的训练周期循环训练
#		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)#训练模型
#		model.reset_states()#在训练时期结束时重置内部状态，为下一次训练迭代做好准备
#	return model#返回最后训练模型的参数


"""batch_size必须设置为1.这是因为它必须是训练和测试数据集大小的因子。

的预测（）的模型函数也由批量大小约束; 它必须设置为1，因为我们有兴趣对测试数据进行一步预测。

我们不会在本教程中调整网络参数; 相反，我们将使用以下配置，通过一些试验和错误找到：

批量大小：1
时代：3000
神经元：4
作为本教程的扩展，您可能希望探索不同的模型参数，看看是否可以提高性能。

更新：考虑尝试1500个纪元和1个神经元，性能可能会更好！
接下来，我们将了解如何使用适合的LSTM模型进行一步预测。"""