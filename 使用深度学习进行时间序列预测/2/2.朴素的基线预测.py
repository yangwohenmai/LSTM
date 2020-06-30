"""
以数据中,臭氧日出现的次数总和,在总天数中占比的比例,
作为某一日可能是臭氧日的概率
"""
# naive prediction method
from sklearn.metrics import brier_score_loss
from numpy import loadtxt
# load datasets
train = loadtxt('train.csv', delimiter=',')
test = loadtxt('test.csv', delimiter=',')
# 最后一列只有0,1两种数值类型，0代表臭氧日，1代表非臭氧日
# 对所有行的最后一列求和，除以总行数，就是所有行中1出现的概率
naive = sum(train[:,-1]) / train.shape[0]
print(naive)
# forecast the test dataset
yhat = [naive for _ in range(len(test))]
# evaluate forecast
testy = test[:, -1]
bs = brier_score_loss(testy, yhat)
print('Brier Score: %.6f' % bs)
# calculate brier skill score
bs_ref = bs
bss = (bs - bs_ref) / (0 - bs_ref)
print('Brier Skill Score: %.6f' % bss)
