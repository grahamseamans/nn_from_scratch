import numpy as np
import idx2numpy
import timeit as time
import matplotlib.pyplot as plt


def layer_process(vect_in, weights, biases):
    out = np.matmul(vect_in, weights)
    out += biases
    # out = sigmoid(out)
    # relu(out)
    return out


def relu(x):
    x *= (x > 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def normalize(x):
    denom = max(x) - min(x)
    if (denom == 0):
        return 0  #np.zeros instead?
    else:
        return (x - min(x)) / denom


def label_array(x):
    y = np.zeros(10)
    y[x] = 1
    return y


def two_d_plot(x, y):
    fig, ax = plt.subplots()
    plt.scatter(x, y, cmap='plasma')


avg_weight_init, std_weight_init = 0, .1
l_rate = .0001
ITERATIONS = np.arange(10)
initialization = 'normal'

# train_images = idx2numpy.convert_from_file('./MNIST/t10k-images-idx3-ubyte')
# train_labels = idx2numpy.convert_from_file('./MNIST/t10k-labels-idx1-ubyte')
test_images = idx2numpy.convert_from_file('./MNIST/train-images-idx3-ubyte')
test_labels = idx2numpy.convert_from_file('./MNIST/train-labels-idx1-ubyte')

if initialization == 'normal':
    weights = np.random.normal(avg_weight_init, std_weight_init, (784, 10))
    biases = np.random.normal(avg_weight_init, std_weight_init, (10, ))
elif initialization == 'zeros':
    weights = np.zeros((784, 10))
    biases = np.zeros((10, ))

plot_data = []

for (vect_in, label, i) in zip(test_images, test_labels, ITERATIONS):
    vect_in = vect_in.flatten()
    label_vect = label_array(label)

    out = layer_process(vect_in, weights, biases)
    out = normalize(out)


    pred = np.argmax(out)
    if (label == pred):
        plot_data.append(1)
    else:
        plot_data.append(0)

    cost = out - label_vect

    weights -= (np.outer(vect_in, cost) * l_rate)
    biases -= (cost * l_rate)

# two_d_plot(ITERATIONS, plot_data)
plt.scatter(ITERATIONS, plot_data, marker='o')
