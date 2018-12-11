from random import randint
from numpy import array
from numpy import argmax
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed

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

# prepare data for the LSTM
def get_data(n_in, n_out):
	# generate random sequence
	sequence = generate_sequence()
	# one hot encode
	encoded = one_hot_encode(sequence)
	# convert to X,y pairs
	X,y = to_supervised(encoded, n_in, n_out)
	return X,y

# define LSTM
n_in = 5
n_out = 5
encoded_length = 100
batch_size = 7
model = Sequential()
model.add(LSTM(20, batch_input_shape=(batch_size, n_in, encoded_length), return_sequences=True, stateful=True))
model.add(TimeDistributed(Dense(encoded_length, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# train LSTM
for epoch in range(500):
	# generate new random sequence
	X,y = get_data(n_in, n_out)
	# fit model for one epoch on this sequence
	model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
	model.reset_states()
# evaluate LSTM
X,y = get_data(n_in, n_out)
yhat = model.predict(X, batch_size=batch_size, verbose=0)
# decode all pairs
for i in range(len(X)):
	print('Expected:', one_hot_decode(y[i]), 'Predicted', one_hot_decode(yhat[i]))