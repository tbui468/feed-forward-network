import numpy as np

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
def cross_entropy_grad(pred, truth, z=0.0):
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

#**************testing theoretical ReLog*******************
def relog_single(z):
    if z <= 0:
        return 0
    else:
        return z

def relog_prime_single(z):
    if z <= 0:
        return 0
    else:
        return 1.0

def relog(z):
    relog_vec = np.vectorize(relog_single)
    return relog_vec(z)

def relog_prime(z):
    relog_prime_vec = np.vectorize(relog_prime_single)
    return relog_prime_vec(z)
    
