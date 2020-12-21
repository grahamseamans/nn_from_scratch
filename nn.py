# nn.py Graham Seamans

import numpy as np
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
        self.layerio = []
        self.read_net()
        self.l_rate = learning_rate

    def read_net(self):
        with open("net_shape.txt") as data:
            for line in data:
                self.layers.append(layer(line.split(), l_rate))
                self.layerio.append(0)
            self.layerio.append(0)

    def print_layers(self):
        for layer in self.layers:
            layer.display()

    def predict(self, vect_in):
        i = 0
        for layer in self.layers:
            self.layerio[i] = np.add(vect_in, self.layerio[i])
            vect_in = layer.process(vect_in)
            i += 1
        self.layerio[i] = np.add(vect_in, self.layerio[i])
        return vect_in

    def train(self, cost, batch_size):
        # avg and reverse layers io
        self.layerio.reverse()
        for i in range(len(self.layerio)):
            self.layerio[i] = np.divide(self.layerio[i], batch_size)

        prev_layer_chain = cost

        for i, layer in enumerate(reversed(self.layers)):
            prev_layer_chain = layer.train(
                prev_layer_chain, batch_size, self.layerio[i], self.layerio[i + 1]
            )

        # wipe for next batch
        for i in range(len(self.layerio)):
            self.layerio[i] = 0


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
        out = np.matmul(vect_in, self.weights)
        out += self.biases
        if self.activation == "relu":
            out[out < 0] = 0
        else:
            out = sigmoid(out)
        return out

    def train(self, prev_layer_chain, batch_size, l_out, l_in):
        # derivative of the activation function ds/dnet
        if self.activation == "relu":
            ds_dnet = l_out
            ds_dnet[ds_dnet <= 0] = 0
            ds_dnet[ds_dnet > 0] = 1
        else:
            ds_dnet = np.multiply(l_out, np.subtract(1, l_out))

        # adding the activation derivative to the derivative chain
        layer_chain = prev_layer_chain * ds_dnet

        # adjust weights / biases using derivative of net and chain
        self.weights -= self.l_rate * np.outer(l_in, layer_chain)
        self.biases -= self.l_rate * layer_chain

        # derivative of net for activation dnet/da
        return np.matmul(self.weights, layer_chain)


def normalize(x):
    return (x - x.mean()) / x.std()


# SETUP ----------------------------------------------------

test_images = pickle.load(open("./processed_data/test_images.p", "rb"))
test_labels = pickle.load(open("./processed_data/test_labels.p", "rb"))
train_images = pickle.load(open("./processed_data/train_images.p", "rb"))
train_labels = pickle.load(open("./processed_data/train_labels.p", "rb"))

"""
profiler = cProfile.Profile()
profiler.enable()
"""

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

"""
profiler.disable()
stats = pstats.Stats(profiler)
stats.strip_dirs().sort_stats("tottime").print_stats(10)
"""
print("correct prediction rate: " + str(correct_pred / len(test_labels)))
