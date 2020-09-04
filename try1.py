import numpy as np
import idx2numpy
import timeit as time


def layer_process(vect_in, weights, biases):
    out = np.matmul(vect_in, weights) + biases
    relu(out)
    return out


def relu(x):
    x *= (x > 0)


def normalize(x):
    return (x - min(x)) / (max(x) - min(x))

def label_array(x):
    y = np.zeros(10)
    y[x] = 1
    return y

avg_weight_init, std_weight_init = 0, .1
l_rate = .0001
ITERATIONS = np.zeros(1000)

# train_images = idx2numpy.convert_from_file('./MNIST/t10k-images-idx3-ubyte')
# train_labels = idx2numpy.convert_from_file('./MNIST/t10k-labels-idx1-ubyte')
test_images = idx2numpy.convert_from_file('./MNIST/train-images-idx3-ubyte')
test_labels = idx2numpy.convert_from_file('./MNIST/train-labels-idx1-ubyte')

y = label_array(test_labels[0])

flattened_test = []

# print(test_labels.shape)

weights = np.random.normal(avg_weight_init, std_weight_init, (784, 10))
biases = np.random.normal(avg_weight_init, std_weight_init, (10, ))

for (vect_in, label, i) in zip(test_images, test_labels, ITERATIONS):
    vect_in = vect_in.flatten()
    label = label_array(label) # replace this with something lighter weight if it's slow
    out = layer_process(vect_in, weights, biases)
    out = normalize(out)

    cost = label - out

    weights = weights - (np.outer(vect_in, cost) * l_rate) # do i need to transpose something???
    biases = biases - (cost * l_rate)
    # print(weights.mean())
    print(cost.mean())



'''

so now... how the heck do we do the gradient descent? 
we know what the loss function is, it's:
sum(ground truth - nets output)
but how do we take the derivative of this across so 
many different parts of a net. like how do we know what the 
loss is of a single weight?

l_rate = leaning rate

so for the weights: w1 = w0 - error * vect_in * l_rate
and for biases: b1 = b0 - error * l_rate

for weights:
weights is [786, 10] error is [10,1] vect in is [786,1] l_rate is scalar

so mult vect in in some weird way by error? 
n vect * m vect = n x m matrix?? 
this is called the outer product, np.outer()
then all we need to do is mult by a scalar
weight1 = weight0 - (np.outer(vect_in, error) * l_rate)

for biases:
bases is [10,1] error is [10,1] l_rate is scalar
bias1 = bias0 - (error * l_rate)
'''



