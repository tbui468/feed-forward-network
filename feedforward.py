import gzip
import pickle
import numpy as np
import random

#look at Nielson book on computing stochastic gradient descent
    #he keeps track of the batch dimension all the way up to the final weight/biases update - there he averages the errors * prev_activations / errors
    #is this any different from averaging it when computing errors????

    #could this be the current problem??? Earlier the problem might have been the wrong weight initialization (all were set to range [0, 1) rather than [-.5, .5),
        #and all the activations were saturating on the first training batch

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
    return np.sum(truth - pred, axis=0) / pred.shape[0]

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
#activations and z need to be 2d matrices since the first dimension is for batch - or just average them
class FNN:
    def __init__(self, dims):
        self.weights = [np.random.rand(input_dim, output_dim) - 0.5 for input_dim, output_dim in zip(dims[:-1], dims[1:])]
        self.biases = [np.zeros((1, x)) for x in dims[1:]]
        self.a = [np.zeros(dim) for dim in dims[1:]]
        self.z = [np.zeros(dim) for dim in dims[1:]]
        self.errors = [np.zeros(dim) for dim in dims[1:]]
   
    def forward(self, x):
        batch_size = x.shape[0]
        y = x
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.matmul(y, w) + np.tile(b, (x.shape[0], 1)) #tiling biases may be unnecessary, it's clearer
            a = sigmoid(z)
            y = a
            #save activations and linear combination (z) for backprop 
            self.z[idx] = np.sum(z, axis=0) / batch_size
            self.a[idx] = np.sum(a, axis=0) / batch_size
        return y

    def train_batch(self, batch):
        lr = 0.1
        x, truth = batch
        pred = self.forward(x)
        batch_size = x.shape[0]
        
        #updating weights/biases in final layer
        cost_grad = mse_prime(pred, truth) #dC/da for all activations in output layer
        sigmoid_p2 = sigmoid_prime(self.z[1]) #dsigma/dz
        self.errors[1] = np.multiply(cost_grad, sigmoid_p2) # 10 x 
        self.weights[1] -= lr * np.matmul(np.reshape(self.a[0], (-1, 1)), np.reshape(self.errors[1], (1, -1)))
        self.biases[1] -= lr * self.errors[1]


        #updating weights and biases in second layer
        weight_err2 = np.matmul(self.weights[1], np.reshape(self.errors[1], (-1, 1)))
        sigmoid_p1 = sigmoid_prime(self.z[0])
        self.errors[0] = np.multiply(weight_err2, np.reshape(sigmoid_p1, (-1, 1)))
        avg_inputs = np.sum(x, axis=0) / batch_size
        self.weights[0] -= lr * np.matmul(np.reshape(avg_inputs, (-1, 1)), np.transpose(self.errors[0]))
        self.biases[0] -= lr * np.transpose(self.errors[0])

