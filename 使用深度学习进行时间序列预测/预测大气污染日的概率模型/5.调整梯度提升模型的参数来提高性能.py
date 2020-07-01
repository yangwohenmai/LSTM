"""
本程序通过对例4中的梯度提升模型调整参数，来提高预测的准确率。
分别调整了深度，学习率，采样集，和树数，通过brier skill score值来评价结果
"""
from numpy import loadtxt
from numpy import mean
from matplotlib import pyplot
from sklearn.base import clone
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# 拟合并评估给定的已定义和配置的scikit-learn模型并返回BSS
def evaluate_once(bs_ref, template, trainX, trainy, testX, testy):
    # 配置模型，用训练值trainX、trainY训练模型
    model = clone(template)
    model.fit(trainX, trainy)
    """
    predict_proba输出的结果可能是n行k列的数组，n是要预测数据的行数，k表示有k种类型的输出，
    如果输出结果有4中类型：A,B,C,D，则k=4，在预测前要将A,B,C,D转化为1,2,3,4表示的数值，
    输出的k列按照类型对应的数值大小从0到n排列，即输出k列中，第1列表示类型1对应的概率，第k列表示4对应的概率
    本质上排序规则是根据self.classes_进行排序：https://www.jb51.net/article/189604.htm
    """
    # 每行第一列表示结果是0(非臭氧日)的概率，第二列表示输出结果是1(臭氧日)的概率
    probs = model.predict_proba(testX)
    # 取probs的第二列数据(臭氧日)作为输出结果
    yhat = probs[:, 1]
    # 计算brier分数（brier score）
    bs = brier_score_loss(testy, yhat)
    # 计算brier skill分数（brier skill score）
    bss = (bs - bs_ref) / (0 - bs_ref)
    return bss

# 重复评估给定sklearn模型n次，计算BSS平均分数，并返回这些分数进行分析
def evaluate(bs_ref, model, trainX, trainy, testX, testy, n=10):
    scores = [evaluate_once(bs_ref, model, trainX, trainy, testX, testy) for _ in range(n)]
    print('>%s, bss=%.6f' % (type(model), mean(scores)))
    return scores

# 计算朴素预测的Brier分数，以便能够计算新模型的BSS
def calculate_naive(train, test, testy):
    # 求出所有数据中1出现的概率naive，也就是臭氧日的概率
    naive = sum(train[:,-1]) / train.shape[0]
    # 预测测试集，其实就是默认每一天是臭氧日的概率都等于naive
    yhat = [naive for _ in range(len(test))]
    # 计算预测结果（朴素概率）和真实结果之间的 bs 分数
    bs_ref = brier_score_loss(testy, yhat)
    return bs_ref

# 加载数据
train = loadtxt('train.csv', delimiter=',')
test = loadtxt('test.csv', delimiter=',')
# 把数据拆分成训练集和测试集，构造 监督学习类型 的输入和输出数据
# 第一天的前71列数据作为输入，对应第二天的最后一列数据作为输出
trainX, trainy, testX, testy = train[:,:-1],train[:,-1],test[:,:-1],test[:,-1]
bs_ref = calculate_naive(train, test, testy)


# 以下部分评估不同参数下集成决策树的结果
scores, names = list(), list()
# 基线
model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, subsample=1.0, max_depth=3)
avg_bss = evaluate(bs_ref, model, trainX, trainy, testX, testy)
scores.append(avg_bss)
names.append('base')
# 较低的学习率和更多的树
model = GradientBoostingClassifier(learning_rate=0.01, n_estimators=500, subsample=1.0, max_depth=3)
avg_bss = evaluate(bs_ref, model, trainX, trainy, testX, testy)
scores.append(avg_bss)
names.append('lr')
# 增加的最大树深度和较小的数据集采样
model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, subsample=0.7, max_depth=7)
avg_bss = evaluate(bs_ref, model, trainX, trainy, testX, testy)
scores.append(avg_bss)
names.append('depth')
# 所有的都调整
model = GradientBoostingClassifier(learning_rate=0.01, n_estimators=500, subsample=0.7, max_depth=7)
avg_bss = evaluate(bs_ref, model, trainX, trainy, testX, testy)
scores.append(avg_bss)
names.append('all')
# 画图
pyplot.boxplot(scores, labels=names)
pyplot.show()