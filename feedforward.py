import gzip
import pickle
import numpy as np
import random

#definitely related to how errors/gradient descent is calculated
#why are the weights/erros in second step in backprop dimensions 8 x 32 (where is the 8 coming from????)

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
    def __init__(self, data, batch_size):
        self.data = data
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

    def empty(self):
        return self.index >= len(self.data)


def one_hot_encode(label):
    vec = np.zeros(10)
    vec[label] = 1.0
    return vec

#********************NETWORK********************************

#***************MSE and Sigmoid activation****************
def mse(pred, truth):
    squared_diff = np.square(truth - pred) / 2.0
    return np.sum(squared_diff, axis=0) / pred.shape[0]

def mse_prime(pred, truth):
    return (truth - pred)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

#************Cross entropy and Softmax (for output)*********
#def cross_entropy(pred, truth):
#    one_minus_pred = np.ones(pred.shape) - pred
#    one_minus_truth = np.ones(truth.shape) - truth
#    term_one = np.multiply(truth, np.log(pred))
#    term_two = np.multiply(one_minus_truth, np.log(one_minus_pred))
#    return -np.sum(term_one + term_two)
#
#
#def cross_entropy_grad(pred, truth):
#    term_one = -np.divide(truth, pred)
#    term_two = np.divide(1 - truth, 1 - pred)
#    return term_one + term_two
#
#
#def softmax(z):
#    exp_sum = np.sum(np.exp(z), axis=1)
#    return np.divide(np.exp(z), np.reshape(exp_sum, (-1, 1)))



#define the vanilla feedforward network
#activations and z need to be 2d matrices since the first dimension is for batch
class FNN:
    def __init__(self, dims):
        self.weights = [np.random.normal(0, 1.0 / np.sqrt(input_dim), (input_dim, output_dim)) for input_dim, output_dim in zip(dims[:-1], dims[1:])]
        self.biases = [np.zeros((1, x)) for x in dims[1:]]
        self.a = [np.zeros(dim) for dim in dims[1:]]
        self.z = [np.zeros(dim) for dim in dims[1:]]
        self.errors = [np.zeros(dim) for dim in dims[1:]]
   
    def forward(self, x):
        y = x
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            self.z[idx] = np.matmul(y, w) + np.tile(b, (x.shape[0], 1)) #tiling biases may be unnecessary, it's clearer
            self.a[idx] = sigmoid(self.z[idx])
            y = self.a[idx]
        return y

    def train_batch(self, batch):
        lr = 1.0
        x, y = batch
        pred = self.forward(x)
        truth = y
        
        #updating weights/biases in final layer
        cost_grad = mse_prime(pred, truth) #dC/da for all activations in output layer
        sigmoid_p2 = sigmoid_prime(self.z[1]) #dsigma/dz
        self.errors[1] = np.multiply(cost_grad, sigmoid_p2) #32 x 10, 32 x 10
        self.weights[1] -= lr * np.matmul(np.transpose(self.a[0]), self.errors[1])  #shouldn't the inner dimension be the batch???
        self.biases[1] -= lr * np.sum(self.errors[1], axis=0)


        #updating weights and biases in second layer
        weight_err2 = np.matmul(self.errors[1], np.transpose(self.weights[1])) #32 x 10, 10 x 32
        sigmoid_p1 = sigmoid_prime(self.z[0])
        self.errors[0] = np.multiply(np.transpose(weight_err2), sigmoid_p1) #what is goind on here?  why is sigmoid 8x32, and weight_err2 8 x 32
        self.weights[0] -= lr * np.matmul(np.transpose(x), self.errors[0])
        self.biases[0] -= lr * np.sum(self.errors[0], axis=0)

