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


def two_d_plot(x, y, title = ''):
    fig, ax = plt.subplots()
    plt.scatter(x, y, cmap='plasma')
    plt.title(title)

def avg_every_x(arr, x):
    return np.mean(arr.reshape(-1, x), axis=1)

avg_weight_init, std_weight_init = 0, .1
l_rate = .0001
ITERATIONS = np.arange(1000)
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

plot_data = np.zeros(ITERATIONS.shape)
correct_pred = 0

# train

for (vect_in, label, i) in zip(test_images, test_labels, ITERATIONS):
    vect_in = vect_in.flatten()
    label_vect = label_array(label)

    out = layer_process(vect_in, weights, biases)
    out = normalize(out)


    pred = np.argmax(out)
    if (label == pred):
        plot_data[i] = 1
        correct_pred += 1
    else:
        plot_data[i] = 0

    cost = out - label_vect

    weights -= (np.outer(vect_in, cost) * l_rate)
    biases -= (cost * l_rate)


plot_data_10_avg = avg_every_x(plot_data, 10)
ITERATIONS_10_avg = avg_every_x(ITERATIONS, 10)

title = 'correct prediction rate: ' + str(correct_pred / ITERATIONS.size)
two_d_plot(ITERATIONS_10_avg, plot_data_10_avg, title)