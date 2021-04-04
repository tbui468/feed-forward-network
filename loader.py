import gzip
import pickle
import numpy as np


#*********************************DATA PROCESSING*******************************

def load_data(percent=1.0):
    f = gzip.open('./../nielson-book/neural-networks-and-deep-learning/data/mnist.pkl.gz', 'rb')
    train_data, valid_data, test_data = pickle.load(f, encoding="latin1")
    f.close()

    #only load percent of data (for quicker iterations while writing code)
    train_len = int(len(train_data[1]) * percent)
    valid_len = int(len(valid_data[1]) * percent)
    test_len = int(len(test_data[1]) * percent)

    #process data into more convenient format
    train_pixels = [np.array(x, dtype=np.float32) for x in train_data[0][:train_len]]
    train_labels = [one_hot_encode(x) for x in train_data[1][:train_len]]

    valid_pixels = [np.array(x, dtype=np.float32) for x in valid_data[0][:valid_len]]
    valid_labels = [one_hot_encode(x) for x in valid_data[1][:valid_len]]

    test_pixels = [np.array(x, dtype=np.float32) for x in test_data[0][:test_len]]
    test_labels = [one_hot_encode(x) for x in test_data[1][:test_len]]

    return list(zip(train_pixels, train_labels)), list(zip(valid_pixels, valid_labels)), list(zip(test_pixels, test_labels))


#data is a list of tuples (pixels, one-hot encoded label)
class Loader:
    def __init__(self, data, batch_size=0):
        self.data = data
        if batch_size == 0: #if no batch size given, use entire dataset
            self.batch_size = len(data)
        else:
            self.batch_size = batch_size
        self.index = 0

    def next(self):
        end = min(self.index + self.batch_size, len(self.data))

        input_data = [i[0] for i in self.data[self.index: end]]
        label_data = [i[1] for i in self.data[self.index: end]]

        self.index = end

        return np.vstack(input_data), np.vstack(label_data)

    def shuffle(self):
        random.shuffle(self.data)

    def reset(self):
        self.index = 0

    def size(self):
        return len(self.data)

    def empty(self):
        return self.index >= len(self.data)


def one_hot_encode(label):
    vec = np.zeros(10)
    vec[label] = 1.0
    return vec

