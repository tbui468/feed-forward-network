#the convoluational network
import numpy as np
import network_utils as nu

#derive backprop equations for convnets (look at JefKine blog bookmarked on this topic)

class CNN:
    def __init__(self):
        #two layer convnet: input -> convolution + pooling (no weights for pooling, right?) -> ReLU -> linear -> softmax
        self.img_dim = 28 #width = height for input image

        #conv layer weights and biases
        self.map_count = 2
        self.conv_kdim = 5
        self.conv_outdim = self.img_dim - self.conv_kdim + 1
        self.pool_kdim = 3
        self.pool_outdim = int(self.conv_outdim / self.pool_kdim) #using a stride of 1 for conv kernel, and stride of 3 for pooling kernel
        np.random.seed(0) 
        self.conv_weights = np.random.normal(0.0, 1.0 / np.sqrt(self.map_count * self.conv_kdim * self.conv_kdim), (self.map_count, self.conv_kdim, self.conv_kdim))
        np.random.seed(0) 
        self.conv_biases = np.zeros((1, self.map_count))
        
        #linear layer weights and biases
        np.random.seed(0) 
        self.linear_weights = np.random.normal(0.0, 1.0 / np.sqrt(self.map_count * self.pool_outdim * self.pool_outdim), (self.map_count * self.pool_outdim * self.pool_outdim, 10))
        np.random.seed(0)
        self.linear_biases = np.zeros((1, 10))

        self.conv_z = None
        self.conv_a = None
        self.linear_z = None
        self.linear_a = None


    #test convolutional layer output first
    def forward(self, x):
        batch_size = x.shape[0]
        self.conv_z = np.zeros((batch_size, self.map_count * self.pool_outdim * self.pool_outdim))
        self.conv_a = np.zeros((batch_size, self.map_count * self.pool_outdim * self.pool_outdim))
        self.linear_z = np.zeros((batch_size, 10)) #batch size x outputs per sample
        self.linear_a = np.zeros((batch_size, 10)) #batch size x outputs per sample
        
        for sample in range(0, batch_size):
            img = x[sample, 0] #(dimension 28 x 28); taking sample 0, channel 0
            f_maps = np.zeros((self.map_count, self.conv_outdim, self.conv_outdim))
            p_maps = np.zeros((self.map_count, self.pool_outdim, self.pool_outdim))

            #perform convolution on test image
            for row in range(0, self.conv_outdim):
                for col in range(0, self.conv_outdim):
                    for map_idx in range(0, self.map_count):
                        lrf = img[row: row + self.conv_kdim, col: col + self.conv_kdim] #local receptive field
                        f_maps[map_idx, row, col] = np.sum(np.multiply(lrf, self.conv_weights[map_idx])) + self.conv_biases[0, map_idx]

            #perform max pooling
            for row in range(0, self.pool_outdim):
                for col in range(0, self.pool_outdim):
                    for map_idx in range(0, self.map_count):
                        conv_map = f_maps[map_idx]
                        lrf = conv_map[row * self.pool_kdim: (row + 1) * self.pool_kdim, col * self.pool_kdim: (col + 1) * self.pool_kdim]
                        p_maps[map_idx, row, col] = np.amax(lrf)

            
            self.conv_z[sample, :] = p_maps.reshape(1, -1) #flatten all pooling maps
            self.conv_a[sample, :] = nu.sigmoid(self.conv_z[sample, :])

            self.linear_z[sample, :] = np.matmul(self.conv_a[sample, :], self.linear_weights) + self.linear_biases
            self.linear_a[sample, :] = nu.sigmoid(self.linear_z[sample, :])

        return self.linear_a


    def train(self, batch, set_size, lr=0.1, lmbda=5.0):
        x, y = batch
        pred = self.forward(x)
        batch_size = x.shape[0]
        weight_decay = (1.0 - lr * lmbda / set_size)

        #compute error in linear layer and update weights and biases
        cost_grad = nu.cross_entropy_grad(pred, y, self.linear_z) #output is a batch x output_dim error array
        linear_error = cost_grad #cross entropy error doesn't have sigmoid prime term multiplied with it
        self.linear_weights *= weight_decay
        self.linear_weights -= lr * np.matmul(np.transpose(self.conv_a), linear_error) / batch_size
        self.linear_biases -= lr * np.sum(linear_error, axis=0) / batch_size

        #compute error in conv + pooling layer, and update weights and biases

