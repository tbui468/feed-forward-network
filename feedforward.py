import numpy as np
import network_utils as nu

#Andrej Kaparthy Tips:
    #initialize bias to mean (if regression).  If positive:negative ration is 1:10, set weights so initial probability output is .1
    #visualize data BEFORE pred = model(x) to make sure data going in is correct
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

#Generalize to allow any number of feedforward layers (currently set to one hidden layer)
#How to abstract to allow putting in convolution/pooling layers?

#current best 97.6% cross entropy cost function, sigmoid output activation, 50 epochs, zero initial biases, weights initialized to mean 0, std 1/sqrt(input dim), lr =0.3, batch=64
#batch size of 32 with lr =0.2 causes both loss function and accuracy to increase near the end
#best .1851 loss, 97.69% accuracy

#pytorch convnet gives 98%+ with no tuning

#********************NETWORK********************************
#define the vanilla feedforward network
#activations and z need to be 2d matrices since the first dimension is for batch - or just average them
class FNN:
    def __init__(self, dims):
        np.random.seed(0) 
        self.weights = [np.random.normal(0.0, 1.0 / np.sqrt(input_dim), (input_dim, output_dim)) for input_dim, output_dim in zip(dims[:-1], dims[1:])]
        np.random.seed(0)
        self.biases = [np.zeros((1, x)) for x in dims[1:]]
        self.a = [0 for dim in dims[:]]
        self.z = [0 for dim in dims[:]]
   
    def forward(self, x):
        batch_size = x.shape[0]
        y = x
        self.a[0] = x
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            self.z[idx + 1] = np.matmul(y, w) + np.tile(b, (x.shape[0], 1))
            self.a[idx + 1] = nu.sigmoid(self.z[idx + 1])
            y = self.a[idx + 1]
        return y

    def train_batch(self, batch, set_size, lr=0.2, lmbda=5.0):
        x, truth = batch
        pred = self.forward(x)
        batch_size = x.shape[0]
        weight_decay = (1.0 - lr * lmbda / set_size)
        
        #updating weights/biases in final layer
        #cost_grad = mse_grad(pred, truth, self.z[1])
        cost_grad = nu.cross_entropy_grad(pred, truth, self.z[-1])
        last_error = cost_grad
        self.weights[-1] *= weight_decay
        self.weights[-1] -= lr * np.matmul(np.transpose(self.a[-2]), last_error) / batch_size
        self.biases[-1] -= lr * np.sum(last_error, axis=0) / batch_size

        #loop from second to last layer to layer 2
        for i in range(2, len(self.weights) - 1):
            weight_err2 = np.matmul(last_error, np.transpose(self.weights[-(i - 1)])) #last index
            sigmoid_p1 = nu.sigmoid_prime(self.z[-i])
            current_error = np.multiply(weight_err2, sigmoid_p1)
            self.weights[-i] *= weight_decay
            self.weights[-i] -= lr * np.matmul(np.transpose(self.a[-(i + 1)]), current_error) / batch_size
            self.biases[-i] -= lr * np.sum(current_error, axis=0) / batch_size
            last_error = current_error

