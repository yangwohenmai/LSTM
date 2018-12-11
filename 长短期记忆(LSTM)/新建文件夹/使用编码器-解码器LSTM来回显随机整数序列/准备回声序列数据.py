from random import randint
from pandas import DataFrame
from pandas import concat
from numpy import *
set_printoptions(threshold=NaN)

# generate a sequence of random numbers in [0, 99]
def generate_sequence(length=25):
	return [randint(0, 99) for _ in range(length)]

# one hot encode sequence
def one_hot_encode(sequence, n_unique=100):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# convert encoded sequence to supervised learning
def to_supervised(sequence, n_in, n_out):
	# create lag copies of the sequence
	df = DataFrame(sequence)
	df = concat([df.shift(n_in-i-1) for i in range(n_in)], axis=1)
	# drop rows with missing values
	df.dropna(inplace=True)
	# specify columns for input and output pairs
	values = df.values
	width = sequence.shape[1]
	X = values.reshape(len(values), n_in, width)
	y = values[:, 0:(n_out*width)].reshape(len(values), n_out, width)
	return X, y

# generate random sequence
sequence = generate_sequence()
print(sequence)
# one hot encode
encoded = one_hot_encode(sequence)
print(encoded)
# convert to X,y pairs
X,y = to_supervised(encoded, 5, 3)
# decode all pairs
for i in range(len(X)):
	print(one_hot_decode(X[i]), '=>', one_hot_decode(y[i]))