# Example of LSTM to learn a sequence
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# create sequence
length = 10
sequence = [i/float(length) for i in range(length)]
print(sequence)
# create X/y pairs
df = DataFrame(sequence)
df = concat([df.shift(1), df], axis=1)
df.dropna(inplace=True)
# convert to LSTM friendly format
values = df.values
X, y = values[:, 0], values[:, 1]
X = X.reshape(len(X), 1, 1)
# 1. define network
model = Sequential()
model.add(LSTM(10, input_shape=(1,1)))
model.add(Dense(1))
# 2. compile network
model.compile(optimizer='adam', loss='mean_squared_error')
# 3. fit network
history = model.fit(X, y, epochs=1000, batch_size=len(X), verbose=0)
# 4. evaluate network
loss = model.evaluate(X, y, verbose=0)
print(loss)
# 5. make predictions
predictions = model.predict(X, verbose=0)
print(predictions[:, 0])