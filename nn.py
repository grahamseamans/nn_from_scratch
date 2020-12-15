# nn.py Graham Seamans

import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

avg_init, std_init = 0, 0.1
l_rate = 0.005


class neural_net:
    def __init__(self):
        self.layers = []
        self.read_net()

    def read_net(self):
        with open("net_shape.txt") as data:
            for line in data:
                self.layers.append(layer([int(a) for a in line.split()]))

    def print_layers(self):
        for layer in self.layers:
            layer.display()

    def predict(self, vect_in):
        for layer in self.layers:
            vect_in = layer.process(vect_in)
        return vect_in

    def train(self, cost):
        prev_layer_chain = cost
        for layer in reversed(self.layers):
            prev_layer_chain = layer.train(prev_layer_chain)


class layer:
    def __init__(self, args):
        self.weights = np.random.normal(avg_init, std_init, (args[0], args[1]))
        self.biases = np.random.normal(avg_init, std_init, (args[1],))
        self.latest_input = None
        self.latest_output = None

    def process(self, vect_in):
        self.latest_input = vect_in
        out = np.matmul(vect_in, self.weights)
        out += self.biases
        out = sigmoid(out)
        self.latest_output = out
        return out

    def train(self, prev_layer_chain):
        # set up current layer
        ds_dnet = np.multiply(self.latest_output, np.subtract(1, self.latest_output))
        layer_chain = prev_layer_chain * ds_dnet

        # adjust weights / biases
        self.weights -= l_rate * np.outer(self.latest_input, layer_chain)
        self.biases -= l_rate * layer_chain

        # setup for next layer
        layer_chain = np.matmul(self.weights, layer_chain)
        return layer_chain

    def display(self):
        print(self.weights)
        print(self.biases)


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


# SETUP ----------------------------------------------------

test_images = idx2numpy.convert_from_file("./MNIST/t10k-images-idx3-ubyte")
test_labels = idx2numpy.convert_from_file("./MNIST/t10k-labels-idx1-ubyte")
train_images = idx2numpy.convert_from_file("./MNIST/train-images-idx3-ubyte")
train_labels = idx2numpy.convert_from_file("./MNIST/train-labels-idx1-ubyte")

avg_init, std_init = 0, 0.1
l_rate = 0.005

net = neural_net()

# TRAIN -------------------------------------------------------

for (vect_in, label) in zip(train_images, train_labels):
    # shape data
    vect_in = vect_in.flatten()
    label_vect = label_array(label)

    # process the data
    out = net.predict(vect_in)

    # gradient descent
    cost = out - label_vect
    net.train(cost)

# TEST -------------------------------------------------------

correct_pred = 0

for (vect_in, label) in zip(test_images, test_labels):
    # shape data
    vect_in = vect_in.flatten()
    label_vect = label_array(label)

    # process data
    out = net.predict(vect_in)

    # get data on pred quality
    pred = np.argmax(out)
    if label == pred:
        correct_pred += 1
    else:
        plot = 0

print("correct prediction rate: " + str(correct_pred / test_labels.size))
