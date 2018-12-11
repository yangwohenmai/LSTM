from numpy import array
from numpy import argmax
from keras.utils import to_categorical
# 定义一个实例
data = [1, 3, 2, 0, 3, 2, 2, 1, 0, 1]
data = array(data)
print(data)
# 编码
encoded = to_categorical(data)
print(encoded)
# 逆转换第一行
inverted = argmax(encoded[0])
print(inverted)