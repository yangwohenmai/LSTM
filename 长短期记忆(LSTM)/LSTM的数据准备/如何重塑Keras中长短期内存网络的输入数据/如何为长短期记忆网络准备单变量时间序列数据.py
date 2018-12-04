from numpy import array

# load...
data = list()
n = 5000
for i in range(n):
    data.append([i + 1, (i + 1) * 10])
data = array(data)
print(data[:5, :])
print(data.shape)

data = data[:, 1]
print(data.shape)
print(data)

samples = list()
length = 200
# step over the 5,000 in jumps of 200
for i in range(0,n,length):
    # grab from i to i + 200
    sample = data[i:i+length]
    samples.append(sample)
print(len(samples))

data = array(samples)
print(data.shape)
print(data)

data = data.reshape((len(samples), length, 1))
print(data.shape)
print(data)