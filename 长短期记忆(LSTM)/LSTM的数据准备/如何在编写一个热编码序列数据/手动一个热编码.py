from numpy import argmax
# 定义一个编码字符串
data = 'hello world'
print(data)
# 定义输入值得所有分类类型
alphabet = 'abcdefghijklmnopqrstuvwxyz '
# define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# 转换成整形编码输入
integer_encoded = [char_to_int[char] for char in data]
print(integer_encoded)
# 编码过程
onehot_encoded = list()
for value in integer_encoded:
	letter = [0 for _ in range(len(alphabet))]
	letter[value] = 1
	onehot_encoded.append(letter)
print(onehot_encoded)
# 逆转换第一行数据
inverted = int_to_char[argmax(onehot_encoded[0])]