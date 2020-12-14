# nn.py Graham Seamans

import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid


def layer_process(vect_in, weights, biases):
    out = np.matmul(vect_in, weights)
    out += biases
    out = sigmoid(out)
    # relu(out)
    return out


def relu(x):
    x *= x > 0


def normalize(x):
    denom = max(x) - min(x)
    if denom == 0:
        return 0
    else:
        return (x - min(x)) / denom


def label_array(x):
    y = np.zeros(10)
    y[x] = 1
    return y


def two_d_plot(x, y, title="", x_label=""):
    fig, ax = plt.subplots()
    plt.scatter(x, y, cmap="plasma")
    plt.title(title)
    plt.xlabel(x_label)
    plt.plot()


def avg_every_x(arr, x):
    return np.mean(arr.reshape(-1, x), axis=1)


# SETUP ----------------------------------------------------

avg_weight_init, std_weight_init = 0, 0.1
l_rate = 0.0001
# ITERATIONS = np.arange(1000)
initialization = "zeroes"

test_images = idx2numpy.convert_from_file("./MNIST/t10k-images-idx3-ubyte")
test_labels = idx2numpy.convert_from_file("./MNIST/t10k-labels-idx1-ubyte")
train_images = idx2numpy.convert_from_file("./MNIST/train-images-idx3-ubyte")
train_labels = idx2numpy.convert_from_file("./MNIST/train-labels-idx1-ubyte")

"""
if initialization == "norm_dist":
    weights = np.random.normal(avg_weight_init, std_weight_init, (784, 10))
    biases = np.random.normal(avg_weight_init, std_weight_init, (10,))
elif initialization == "zeroes":
"""
weights2 = np.zeros((784, 20))
biases2 = np.zeros((20,))
weights1 = np.zeros((20, 10))
biases1 = np.zeros((10,))


# TRAIN -------------------------------------------------------

plot_data = np.zeros_like(train_labels)
correct_pred = 0
index = 0

for (vect_in, label) in zip(train_images, train_labels):
    # shape data
    vect_in = vect_in.flatten()
    label_vect = label_array(label)

    # process the data
    out2 = layer_process(vect_in, weights2, biases2)
    out1 = layer_process(out2, weights1, biases1)
    out1 = normalize(out1)
    cost = out - label_vect

    # get data on learning quality
    pred = np.argmax(out)
    if label == pred:
        plot_data[index] = 1
        correct_pred += 1
    else:
        # replace below with me?
        # plot_data[index] = 0
        plot = 0
    index += 1

    """
    dE/db1 = dE/ds*ds/dnet*dnet/db1
    dE/dw1 = dE/ds*ds/dnet*dnet/dw1
    dE/db2 = dE/ds*ds/dnet*dnet/ds*ds/dnet*dnet/db2
    dE/dw2 = dE/ds*ds/dnet*dnet/ds*ds/dnet*dnet/dw2

    out_x = output of layer x
    out = output of net
    w_x = weights of layer x
    in_x = input of layer x
    label = labeled data, target

    so the terms are:
    dE/ds = -(label - out), - cost
    ds/dnet = out_x(1 - out_x)
    dnet/ds = w_x
    dnet/dwx = in_x
    dnet/dbx = 1

    """
    # gradient descent

    # dE_dnet = cost * out1(1 - out1)
    weights1 -= l_rate * np.outer(out2, cost * out1(1 - out1))
    biases1 -= l_rate * cost * out1(1 - out1)
    
    
    #out = np.matmul(vect_in, weights)
    dE_ds2 = np.matmul(weights1, cost * out1(1 - out1))

    weights2 -= l_rate * np.outer(out2, dE_ds2 * out2(1-out2))
    biases2 -= l_rate * cost

plot_data_avg = avg_every_x(plot_data, 100)
x_axis = np.arange(0, plot_data.size, 100)

two_d_plot(x_axis, plot_data_avg, "learning", "test iter")

# TEST -------------------------------------------------------

plot_data = np.zeros_like(test_labels)
correct_pred = 0
index = 0

for (vect_in, label) in zip(test_images, test_labels):
    # shape data
    vect_in = vect_in.flatten()
    label_vect = label_array(label)

    # process data
    out = layer_process(vect_in, weights, biases)
    out = normalize(out)

    # get data on pred quality
    pred = np.argmax(out)
    if label == pred:
        plot_data[index] = 1
        correct_pred += 1
    else:
        plot = 0

    index += 1

plot_data_avg = avg_every_x(plot_data, 100)
x_axis = np.arange(0, plot_data.size, 100)

title = "correct prediction rate: " + str(correct_pred / test_labels.size)
two_d_plot(x_axis, plot_data_avg, title, "test iter")

print(title)
