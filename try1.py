import numpy as np
import idx2numpy
import timeit as time


def layer_process(vect_in, weights, biases):
    out = np.matmul(vect_in, weights1)
    out += biases1
    relu(out)
    return out


def relu(x):
    x *= (x > 0)


def normalize(x):
    return (x - min(x)) / (max(x) - min(x))


mu, sigma = 0, .1
# train_images = idx2numpy.convert_from_file('./MNIST/t10k-images-idx3-ubyte')
# train_labels = idx2numpy.convert_from_file('./MNIST/t10k-labels-idx1-ubyte')
test_images = idx2numpy.convert_from_file('./MNIST/train-images-idx3-ubyte')
test_labels = idx2numpy.convert_from_file('./MNIST/train-labels-idx1-ubyte')

flattened_test = []

for row in test_images:
    flattened_test.append(row.flatten())

print(test_labels.shape)

weights1 = np.random.normal(mu, sigma, (784, 10))
biases1 = np.random.normal(mu, sigma, (10, ))

out = layer_process(flattened_test[0], weights1, biases1)
out = normalize(out)
print(out)


