import feedforward as fnn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#pytorch version works using custom dataloader
#using sigmoid for activation, and cross entropy or mean-squared-error for loss function

#now look at my custom one and test again
#wrap scripts into two functions (train pytorch model vs train custom model)


train_data, valid_data, test_data = fnn.load_data(0.5)


batch_size = 32
epochs = 30

train_loader = fnn.Loader(train_data, batch_size)
valid_loader = fnn.Loader(valid_data, batch_size)
test_loader = fnn.Loader(test_data, batch_size)


def custom_net():
    layer_dims = [784, 32, 10]
    network = fnn.FNN(layer_dims)
    
    #network.train_batch(train_loader.next())
    #print(network.forward(train_loader.next()[0]))

    for epoch in range(epochs):
        train_loader.reset()
        train_loader.shuffle()
        while not train_loader.empty():
            network.train_batch(train_loader.next())

        valid_loader.reset()
        valid_loader.shuffle()
        total = 0
        correct = 0
        loss = 0

        while not valid_loader.empty():
            x, labels = valid_loader.next()
            pred = network.forward(x)
            loss += fnn.mse(pred, labels)
            for a, b in zip(np.argmax(labels, axis=1), np.argmax(pred, axis=1)):
                if a == b:
                    correct += 1
                total += 1
        print("Loss: ", loss, " Correct: ", correct, "/", total)

custom_net()

def pytorch_net():
    #define a ff network in pytorch
    class TestNet(nn.Module):
        def __init__(self):
            super(TestNet, self).__init__()
            self.fc1 = nn.Linear(784, 32)
            self.fc2 = nn.Linear(32, 10)
            self.sigmoid = nn.Sigmoid()

        def forward(self, batch):
            layer1 = self.sigmoid(self.fc1(batch))
            layer2 = self.sigmoid(self.fc2(layer1))
            return layer2
            

    network = TestNet()

    optimizer = optim.SGD(network.parameters(), lr=0.1)
    #criterion = nn.MSELoss() #note: prediction and output are both one-hot encoded
    criterion = nn.CrossEntropyLoss() #note: doesn't expect one-hot encoded vectors for target!!! (prediction is one-hot encoded)
    #other option: use softmax (instead of sigmoid) on output, and then use log-likelihood for loss function

    for epoch in range(epochs):
        train_loader.reset()
        train_loader.shuffle()
        network.train()
        while not train_loader.empty():
            #network.train_batch(train_loader.next())
            optimizer.zero_grad()
            x, labels = train_loader.next()
            x = torch.Tensor(x)
            labels = torch.Tensor(labels)
            pred = network(x)
            loss = criterion(pred, torch.argmax(labels, axis=1))
            loss.backward()
            optimizer.step()

        valid_loader.reset()
        valid_loader.shuffle()
        total = 0
        correct = 0
        loss = 0
        network.eval()

        while not valid_loader.empty():
            x, labels = valid_loader.next()
            x = torch.Tensor(x)
            labels = torch.Tensor(labels)
            with torch.no_grad():
                pred = network(x)
                loss += criterion(pred, torch.argmax(labels, axis=1))
                for a, b in zip(torch.argmax(labels, axis=1), torch.argmax(pred, axis=1)):
                    if a == b:
                        correct += 1
                    total += 1
        print("Loss: ", loss.item(), " Correct: ", correct, "/", total)
