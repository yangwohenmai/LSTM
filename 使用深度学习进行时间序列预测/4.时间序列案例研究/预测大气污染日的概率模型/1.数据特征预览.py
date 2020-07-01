# load and summarize
from pandas import read_csv
from matplotlib import pyplot
# load dataset
data = read_csv('eighthr.data', header=None, index_col=0, parse_dates=True, squeeze=True)
print(data.shape)
# summarize class counts
counts = data.groupby(73).size()
for i in range(len(counts)):
    percent = counts[i] / data.shape[0] * 100
    print('Class=%d, total=%d, percentage=%.3f' % (i, counts[i], percent))
