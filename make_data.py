import numpy as np
import idx2numpy
import pickle
import os

def label_array(x):
    y = np.zeros(10)
    y[x] = 1
    return y

try:  
    os.mkdir("./processed_data")  
except OSError as error:  
    # hopefully only the error the the dir existing.
    pass

test_images_read = idx2numpy.convert_from_file("./MNIST/t10k-images-idx3-ubyte")
test_labels_read = idx2numpy.convert_from_file("./MNIST/t10k-labels-idx1-ubyte")
train_images_read = idx2numpy.convert_from_file("./MNIST/train-images-idx3-ubyte")
train_labels_read = idx2numpy.convert_from_file("./MNIST/train-labels-idx1-ubyte")

test_images = []
train_labels = []
train_images = []

for i in range(len(test_images_read)):
    test_images.append(test_images_read[i].flatten())

for i in range(len(train_images_read)):
    train_images.append(train_images_read[i].flatten())
    train_labels.append(label_array(train_labels_read[i]))

pickle.dump(test_images, open("./processed_data/test_images.p", "wb"))
pickle.dump(test_labels_read, open("./processed_data/test_labels.p", "wb"))
pickle.dump(train_images, open("./processed_data/train_images.p", "wb"))
pickle.dump(train_labels,  open("./processed_data/train_labels.p", "wb"))
