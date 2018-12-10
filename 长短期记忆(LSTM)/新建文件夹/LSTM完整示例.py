from pandas import DataFrame
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# binary encode an input pattern, return a list of binary vectors
def encode(pattern, n_unique):
	encoded = list()
	for value in pattern:
		row = [0.0 for x in range(n_unique)]
		row[value] = 1.0
		encoded.append(row)
	return encoded

# create input/output pairs of encoded vectors, returns X, y
def to_xy_pairs(encoded):
	X,y = list(),list()
	for i in range(1, len(encoded)):
		X.append(encoded[i-1])
		y.append(encoded[i])
	return X, y

# convert sequence to x/y pairs ready for use with an LSTM
def to_lstm_dataset(sequence, n_unique):
	# one hot encode
	encoded = encode(sequence, n_unique)
	# convert to in/out patterns
	X,y = to_xy_pairs(encoded)
	# convert to LSTM friendly format
	dfX, dfy = DataFrame(X), DataFrame(y)
	lstmX = dfX.values
	lstmX = lstmX.reshape(lstmX.shape[0], 1, lstmX.shape[1])
	lstmY = dfy.values
	return lstmX, lstmY

# define sequences
seq1 = [3, 0, 1, 2, 3]
seq2 = [4, 0, 1, 2, 4]
# convert sequences into required data format
n_unique = len(set(seq1 + seq2))
seq1X, seq1Y = to_lstm_dataset(seq1, n_unique)
seq2X, seq2Y = to_lstm_dataset(seq2, n_unique)
# define LSTM configuration
n_neurons = 20
n_batch = 1
n_epoch = 250
n_features = n_unique
# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, batch_input_shape=(n_batch, 1, n_features), stateful=True))
model.add(Dense(n_unique, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
# train LSTM
for i in range(n_epoch):
	model.fit(seq1X, seq1Y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
	model.reset_states()
	model.fit(seq2X, seq2Y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
	model.reset_states()

# test LSTM on sequence 1
print('Sequence 1')
result = model.predict_classes(seq1X, batch_size=n_batch, verbose=0)
model.reset_states()
for i in range(len(result)):
	print('X=%.1f y=%.1f, yhat=%.1f' % (seq1[i], seq1[i+1], result[i]))

# test LSTM on sequence 2
print('Sequence 2')
result = model.predict_classes(seq2X, batch_size=n_batch, verbose=0)
model.reset_states()
for i in range(len(result)):
	print('X=%.1f y=%.1f, yhat=%.1f' % (seq2[i], seq2[i+1], result[i]))