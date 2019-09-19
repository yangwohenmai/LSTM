from numpy import array

# load...
data = list()
n = 50
for i in range(n):
    data.append([i + 1, (i + 1) * 10])
data = array(data)
print(data[:5, :])
print(data.shape)

data = data[:, 1]
print(data.shape)
print(data)

samples = list()
length = 2
# step over the 50 in jumps of 2
# length表示每次循环跳跃步长为2
for i in range(0,n,length):
    # grab from i to i + 2
    sample = data[i:i+length]
    samples.append(sample)
print(len(samples))

# (25，2)
data = array(samples)
print(data.shape)
print(data)

# 重塑数据为(25,2,1)， 25个样本，每个样本步长为2，1个特征变量
data = data.reshape((len(samples), length, 1))
print(data.shape)
print(data)