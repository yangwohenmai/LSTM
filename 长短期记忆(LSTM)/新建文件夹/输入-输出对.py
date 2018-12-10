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

seq1 = [3, 0, 1, 2, 3]
encoded = encode(seq1, 5)
X, y = to_xy_pairs(encoded)
for i in range(len(X)):
	print(X[i], y[i])