import convolutional as cnn
import loader as ldr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

train_data, valid_data, test_data = ldr.load_data(1.0)

batch_size = 32
epochs = 10

train_loader = ldr.Loader(train_data, batch_size)
valid_loader = ldr.Loader(valid_data)
test_loader = ldr.Loader(test_data, batch_size)

def pytorch_net():
    #define a ff network in pytorch
    class TestNet(nn.Module):
        def __init__(self):
            super(TestNet, self).__init__()
            self.conv = nn.Conv2d(1, 10, 5)#1 input channel, 5 output channels (feature map), 5x5 kernel, stride = 1 by default
            self.pool = nn.MaxPool2d(3, stride=3) #output of maxpool should be 5 x 8 x 8
            self.fc = nn.Linear(10 * 8 * 8, 10)
            self.softmax = nn.Softmax(dim=1)
            self.ReLU = nn.ReLU()
            self.flatten = nn.Flatten()

        def forward(self, batch):
            conv_layer = self.ReLU(self.pool(self.conv(batch))) #expect 2d matrix
            flattened = self.flatten(conv_layer)
            linear_layer = self.softmax(self.fc(flattened))
            return linear_layer
            

    network = TestNet()

    optimizer = optim.SGD(network.parameters(), lr=0.9)
    #criterion = nn.MSELoss() #note: prediction and output are both one-hot encoded
    criterion = nn.CrossEntropyLoss() #note: doesn't expect one-hot encoded vectors for target!!! (prediction is one-hot encoded)
    #other option: use softmax (instead of sigmoid) on output, and then use log-likelihood for loss function

    for epoch in range(epochs):
        train_loader.reset()
        network.train()
        while not train_loader.empty():
            #network.train_batch(train_loader.next())
            optimizer.zero_grad()
            x, labels = train_loader.next()
            batch_size = x.shape[0]
            x = torch.Tensor(x)
            labels = torch.Tensor(labels)
            pred = network(x.view(batch_size, 1, 28, 28))
            loss = criterion(pred, torch.argmax(labels, axis=1))
            loss.backward()
            optimizer.step()

        valid_loader.reset()
        total = 0
        correct = 0
        loss = 0
        network.eval()

        while not valid_loader.empty():
            x, labels = valid_loader.next()
            batch_size = x.shape[0]
            x = torch.Tensor(x)
            labels = torch.Tensor(labels)
            with torch.no_grad():
                pred = network(x.view(batch_size, 1, 28, 28))
                loss += criterion(pred, torch.argmax(labels, axis=1))
                for a, b in zip(torch.argmax(labels, axis=1), torch.argmax(pred, axis=1)):
                    if a == b:
                        correct += 1
                    total += 1
        print("Loss: ", loss.item(), " Correct: ", correct, "/", total)


pytorch_net()
