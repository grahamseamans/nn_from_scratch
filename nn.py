# nn.py Graham Seamans

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
import cProfile
import pstats
import pickle


avg_init, std_init = 0, 0.1
l_rate = 0.001
batch_size = 10
epochs = 5


class neural_net:
    def __init__(self, learning_rate):
        self.layers = []
        self.read_net()
        self.l_rate = learning_rate

    def read_net(self):
        with open("net_shape.txt") as data:
            for line in data:
                self.layers.append(layer(line.split(), l_rate))

    def print_layers(self):
        for layer in self.layers:
            layer.display()

    def predict(self, vect_in):
        for layer in self.layers:
            vect_in = layer.process(vect_in)
        return vect_in

    def train(self, cost, batch_size):
        prev_layer_chain = cost
        for layer in reversed(self.layers):
            prev_layer_chain = layer.train(prev_layer_chain, batch_size)


class layer:
    def __init__(self, args, l_rate):
        in_size = int(args[0])
        out_size = int(args[1])

        self.activation = args[2]
        self.weights = np.random.normal(avg_init, std_init, (in_size, out_size))
        self.biases = np.random.normal(avg_init, std_init, (out_size,))
        self.l_rate = l_rate

        self.latest_input = 0
        self.latest_output = 0

    def process(self, vect_in):
        self.latest_input = np.add(vect_in, self.latest_input)
        out = np.matmul(vect_in, self.weights)
        out += self.biases

        if self.activation == "relu":
            out[out < 0] = 0
        else:
            out = sigmoid(out)

        self.latest_output = np.add(out, self.latest_output)
        return out

    def train(self, prev_layer_chain, batch_size):
        # average inputs and outputs for batches
        self.latest_input = np.divide(self.latest_input, batch_size)
        self.latest_output = np.divide(self.latest_output, batch_size)

        # set up current layer
        # commented out the derivative for the activation funciton
        # predictions are worse worse with it.
        # ds_dnet = np.multiply(self.latest_output, np.subtract(1, self.latest_output))
        layer_chain = prev_layer_chain # * ds_dnet

        # adjust weights / biases
        self.weights -= self.l_rate * np.outer(self.latest_input, layer_chain)
        self.biases -= self.l_rate * layer_chain

        # setup for next layer
        layer_chain = np.matmul(self.weights, layer_chain)

        # reset inputs and ouputs
        self.latest_input = 0
        self.latest_output = 0

        return layer_chain


def normalize(x):
    return (x - x.mean()) / x.std()


def two_d_plot(x, y, title="", x_label=""):
    fig, ax = plt.subplots()
    plt.scatter(x, y, cmap="plasma")
    plt.title(title)
    plt.xlabel(x_label)
    plt.plot()


# SETUP ----------------------------------------------------

test_images = pickle.load(open("./processed_data/test_images.p", "rb"))
test_labels = pickle.load(open("./processed_data/test_labels.p", "rb"))
train_images = pickle.load(open("./processed_data/train_images.p", "rb"))
train_labels = pickle.load(open("./processed_data/train_labels.p", "rb"))

profiler = cProfile.Profile()
profiler.enable()

net = neural_net(l_rate)

# TRAIN -------------------------------------------------------

cost = 0
counter = 0
for _ in range(epochs):
    for (vect_in, label_vect) in zip(train_images, train_labels):
        # process the data
        out = net.predict(vect_in)

        # costs
        cost += out - label_vect

        # gradient descent
        if counter == batch_size:
            cost = np.divide(cost, batch_size)
            net.train(cost, batch_size)
            cost = 0
            counter = 0

        counter += 1

# TEST -------------------------------------------------------

correct_pred = 0

for (input_vector, label) in zip(test_images, test_labels):
    # process data
    out = net.predict(input_vector)

    # get data on pred quality
    pred = np.argmax(out)

    if label == pred:
        correct_pred += 1

profiler.disable()
stats = pstats.Stats(profiler)
stats.strip_dirs().sort_stats("tottime").print_stats(10)

print("correct prediction rate: " + str(correct_pred / len(test_labels)))
