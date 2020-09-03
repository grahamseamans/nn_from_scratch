
'''
print('weights1 shape')
print(weights1.shape)
print('weights')
print(weights1)

print('biases1 shape')
print(biases1.shape)
print('weights')
print(biases1)

print('length flattened pic')
print(len(flattened_test[0]))

out = np.matmul(flattened_test[0], weights1)
print('weighted')
print(out)
out = out + biases1
print('biased')
print(out)

relu(out)
print('reLued')
print(out)
'''
