import numpy as np
import idx2numpy
import timeit as time
import matplotlib.pyplot as plt
from scipy.special import expit


def layer_process(vect_in, weights, biases):
    out = np.matmul(vect_in, weights)
    out += biases
    out = sigmoid(out)
    #relu(out)
    return out


def relu(x):
    x *= (x > 0)


def sigmoid(x):
    return expit(x)


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


def two_d_plot(x, y, title='', x_label=''):
    fig, ax = plt.subplots()
    plt.scatter(x, y, cmap='plasma')
    plt.title(title)
    plt.xlabel(x_label)


def avg_every_x(arr, x):
    return np.mean(arr.reshape(-1, x), axis=1)


avg_weight_init, std_weight_init = 0, .1
l_rate = .0001
# ITERATIONS = np.arange(1000)
initialization = 'normal'

test_images = idx2numpy.convert_from_file('./MNIST/t10k-images-idx3-ubyte')
test_labels = idx2numpy.convert_from_file('./MNIST/t10k-labels-idx1-ubyte')
train_images = idx2numpy.convert_from_file('./MNIST/train-images-idx3-ubyte')
train_labels = idx2numpy.convert_from_file('./MNIST/train-labels-idx1-ubyte')

if initialization == 'normal':
    weights = np.random.normal(avg_weight_init, std_weight_init, (784, 10))
    biases = np.random.normal(avg_weight_init, std_weight_init, (10, ))
elif initialization == 'zeros':
    weights = np.zeros((784, 10))
    biases = np.zeros((10, ))

# train

plot_data = np.zeros_like(train_labels)
correct_pred = 0
index = 0

for (vect_in, label) in zip(train_images, train_labels):
    #shape data
    vect_in = vect_in.flatten()
    label_vect = label_array(label)

    # process the data
    out = layer_process(vect_in, weights, biases)
    out = normalize(out)
    cost = out - label_vect

    #get data on learning quality
    pred = np.argmax(out)
    if (label == pred):
        plot_data[index] = 1
        correct_pred += 1
    else:
        plot = 0
    index += 1

    # gradient descent
    weights -= (np.outer(vect_in, cost) * l_rate)
    biases -= (cost * l_rate)

plot_data_avg = avg_every_x(plot_data, 100)
x_axis = np.arange(0, plot_data.size, 100)

two_d_plot(x_axis, plot_data_avg, 'learning', 'test iter')

# test

plot_data = np.zeros_like(test_labels)
correct_pred = 0
index = 0

for (vect_in, label) in zip(test_images, test_labels):
    #shape data
    vect_in = vect_in.flatten()
    label_vect = label_array(label)

    #process data
    out = layer_process(vect_in, weights, biases)
    out = normalize(out)

    #get data on pred quality
    pred = np.argmax(out)
    if (label == pred):
        plot_data[index] = 1
        correct_pred += 1
    else:
        plot = 0

    index += 1

plot_data_avg = avg_every_x(plot_data, 100)
x_axis = np.arange(0, plot_data.size, 100)

title = 'correct prediction rate: ' + str(correct_pred / test_labels.size)
two_d_plot(x_axis, plot_data_avg, title, 'test iter')

print(title)