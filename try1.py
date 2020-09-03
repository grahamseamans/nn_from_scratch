import numpy as np
import idx2numpy 
import timeit as time

mu, sigma = 0, .1

def layer_process(vect_in, weights, biases):
    out = np.matmul(vect_in, weights1)
    out += biases1
    relu(out)
    return out

def relu(x):
    x *= (x > 0)

# train_images = idx2numpy.convert_from_file('./MNIST/t10k-images-idx3-ubyte')
# train_labels = idx2numpy.convert_from_file('./MNIST/t10k-labels-idx1-ubyte')
test_images = idx2numpy.convert_from_file('./MNIST/train-images-idx3-ubyte')
test_labels = idx2numpy.convert_from_file('./MNIST/train-labels-idx1-ubyte')


flattened_test = []
for row in test_images:
    flattened_test.append(row.flatten())


print(test_labels.shape)

#weights1 = np.ones((784, 10))
weights1 = np.random.normal(mu, sigma, (784, 10))
biases1 = np.random.normal(mu, sigma, (10,))

out = layer_process(flattened_test[0], weights1, biases1)
print(out)




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



