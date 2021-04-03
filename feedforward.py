import gzip
import pickle
import numpy as np
import random

#initialize bias to mean (if regression).  If positive:negative ration is 1:10, set weights so initial probability output is .1
#visualize data BEFORE pred = model(x) 
#test loss over entire valid/test set (not just a batch)
    #add sig figs!!! so that we know if the difference is significant or just noise
#try overfitting a single small batch (as little as two samples) to make sure gradient descent is working (should be able to get to 0)
#consider writing looopy loops (instead of having batches) initially and vectorize after it starts to work

#overfit first with a good model to make loss as small as possible, then give up some of that loss to regularize (to improve validation accuracy)
#use simplilist architecture that will work and get it working
#use Adam 3e-4 (instead of SGD, except on mabye ConvNets)

#dropout
#more data, or augment data
#use pretrained network (if one that fits your goals is available)
#smaller batch gives similar effect to regularization
#early stopping

#use constant lr (rather than learning rate decay) at the beginning.  Tune that after everything works
#use ensembles (average output over group ~5 neural networks).  Can give up to 2% improvement!
#just train longer (learning may seem to stall only to pick up again later)

#change SGD to Adam (adaptive moment estimation)
#if using softmax activation for output, need to define log-likelihood loss and gradients
#add regularization once we find a really low cost

#current best 97.6% cross entropy cost function, sigmoid output activation, 50 epochs, zero initial biases, weights initialized to mean 0, std 1/sqrt(input dim), lr =0.3, batch=64
#batch size of 32 with lr =0.2 causes both loss function and accuracy to increase near the end
#best .1851 loss, 97.69% accuracy

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

#********************NETWORK********************************

#***************MSE/Cross Entropy and Sigmoid activation****************
#MSE is output minus labeled ground truth
def mse(pred, truth):
    batch_size = pred.shape[0]
    return np.square(pred - truth) / 2.0 / batch_size

def mse_grad(pred, truth, z):
    sp = sigmoid_prime(z)
    return np.multiply(pred - truth, sp)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def cross_entropy(pred, truth):
    batch_size = pred.shape[0]
    one_minus_pred = np.ones(pred.shape) - pred
    one_minus_truth = np.ones(truth.shape) - truth
    term_one = np.multiply(truth, np.log(pred))
    term_two = np.multiply(one_minus_truth, np.log(one_minus_pred))
    return -(term_one + term_two) / batch_size

#doesn't use z term, but just keeping consistent APIs for gradient functions
def cross_entropy_grad(pred, truth, z):
    return pred - truth

#************Log likelihood and Softmax (for output)*********
#does using softmax for output change the cross entropy gradient - it might - need to to find derivative wrt input z
def softmax(z):
    exp_sum = np.reshape(np.sum(np.exp(z), axis=1), (-1, 1))
    return np.divide(np.exp(z), np.repeat(exp_sum, z.shape[1], axis=1))

def log_likelihood(pred, truth):
    print('hi')

def log_likelihood_grad(pred, truth, z):
    print('hi')

#define the vanilla feedforward network
#activations and z need to be 2d matrices since the first dimension is for batch - or just average them
class FNN:
    def __init__(self, dims):
        np.random.seed(0) 
        self.weights = [np.random.normal(0.0, 1.0 / np.sqrt(input_dim), (input_dim, output_dim)) for input_dim, output_dim in zip(dims[:-1], dims[1:])]
        np.random.seed(0)
        self.biases = [np.zeros((1, x)) for x in dims[1:]]
        self.a = [0 for dim in dims[1:]]
        self.z = [0 for dim in dims[1:]]
        self.errors = [np.zeros(dim) for dim in dims[1:]]
   
    def forward(self, x):
        batch_size = x.shape[0]
        y = x
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            self.z[idx] = np.matmul(y, w) + np.tile(b, (x.shape[0], 1))
            if idx == 1:
                self.a[idx] = sigmoid(self.z[idx])
            else:
                self.a[idx] = sigmoid(self.z[idx])
            y = self.a[idx]
        return y

    def train_batch(self, batch, set_size):
        lr = 0.2
        lmda = 5.0
        x, truth = batch
        pred = self.forward(x)
        batch_size = x.shape[0]
        weight_decay = (1.0 - lr * lmda / set_size)
        
        #updating weights/biases in final layer
        #cost_grad = mse_grad(pred, truth, self.z[1])
        cost_grad = cross_entropy_grad(pred, truth, self.z[1])
        self.errors[1] = cost_grad
        self.weights[1] *= weight_decay
        self.weights[1] -= lr * np.matmul(np.transpose(self.a[0]), self.errors[1]) / batch_size
        self.biases[1] -= lr * np.sum(self.errors[1], axis=0) / batch_size

        #updating weights and biases in second layer - note: cross entropy cost only avoids sigmoid saturation on output, sigmoid prime terms still in other layers
        weight_err2 = np.matmul(self.errors[1], np.transpose(self.weights[1]))
        sigmoid_p1 = sigmoid_prime(self.z[0])
        self.errors[0] = np.multiply(weight_err2, sigmoid_p1)
        self.weights[0] *= weight_decay
        self.weights[0] -= lr * np.matmul(np.transpose(x), self.errors[0]) / batch_size
        self.biases[0] -= lr * np.sum(self.errors[0], axis=0) / batch_size

