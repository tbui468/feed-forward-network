#PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
#892,3,"Kelly, Mr. James",male,34.5,0,0,330911,7.8292,,Q

import csv
import numpy as np

#ignoreing PassengerId, Name, and Ticket for 


def load_passenger_data(file_path, load_id=False):
    data = []
    with open(file_path, mode='r') as my_file:
        csv_reader = csv.DictReader(my_file)
        #count = sum(1 for row in csv_reader) #call this seems to move iterator to end of file
        max_age = 0
        max_fare = 0
        for row in csv_reader:
            Pclass = -1.
            if 'Pclass' in row:
                Pclass = int(row['Pclass'])

            Sex = -1.
            if 'Sex' in row:
                if row['Sex'] == 'female':
                    Sex = 1.
                else:
                    Sex = 0.

            Age = -1. 
            if 'Age' in row:
                if row['Age'] != '': 
                    Age = float(row['Age'])
                    if Age > max_age:
                        max_age = Age

            SibSp = -1
            if 'SibSp' in row:
                SibSp = int(row['SibSp'])

            Parch = -1
            if 'Parch' in row:
                Parch = int(row['Parch'])

            Fare = -1.
            if 'Fare' in row:
                if row['Fare'] != '': 
                    Fare = float(row['Fare'])
                    if Fare > max_fare:
                        max_fare = Fare

            x = (Pclass, Sex, Age, SibSp, Parch, Fare)


            if load_id:
                y = int(row['PassengerId'])
            else:
                y = [0., 0.]
                if 'Survived' in row:
                    y[int(row['Survived'])] = 1. #one-hot encoding [1, 0] dead, [0, 1] survived, [0, 0] not specified

            data.append((x, y))
        #normalize age and fare here 
        normalized_data = []
        for x, y in data:
            age = x[2] / max_age
            fare = x[5] / max_fare
            normalized_data.append(((x[0], x[1], age, x[3], x[4], fare), y))
    return normalized_data

train_valid_data = load_passenger_data('train.csv')
test_data = load_passenger_data('test.csv', True)

import feedforward as fnn

train_count = int(.999 * len(train_valid_data))

batch_size = 128
train_loader = fnn.Loader(train_valid_data[:train_count], batch_size)
valid_loader = fnn.Loader(train_valid_data[train_count:])
test_loader = fnn.Loader(test_data)


epochs = 2400 #best results were at 800


def custom_net():
    layer_dims = [6, 32, 2]
    network = fnn.FNN(layer_dims)
    
    for epoch in range(epochs):
        train_loader.reset()
        while not train_loader.empty():
            network.train_batch(train_loader.next(), train_loader.size(), lr=0.08, lmbda=0.0)

       
        if epoch % 10 == 9:
            valid_loader.reset()
            total = 0
            correct = 0
            loss = 0
            #why isn't the entire validation set being used???
            while not valid_loader.empty():
                x, labels = valid_loader.next()
                pred = network.forward(x)
                loss += fnn.cross_entropy(pred, labels)
                for a, b in zip(np.argmax(labels, axis=1), np.argmax(pred, axis=1)):
                    if a == b:
                        correct += 1
                    total += 1

            print("valid Loss: ", np.sum(loss), " Correct: ", correct, "/", total)

       
#        if epoch % 10 == 9:
#            train_loader.reset()
#            total = 0
#            correct = 0
#            loss = 0
#            while not train_loader.empty():
#                x, labels = train_loader.next()
#                pred = network.forward(x)
#                loss += np.sum(fnn.cross_entropy(pred, labels), axis=0)
#                for a, b in zip(np.argmax(labels, axis=1), np.argmax(pred, axis=1)):
#                    if a == b:
#                        correct += 1
#                    total += 1
#
#            print("Test Loss: ", np.sum(loss), " Correct: ", correct, "/", total)

    print('done')

    while not test_loader.empty():
        x, labels = test_loader.next()
        pred = network.forward(x)
        rate = np.argmax(pred, axis=1)
        total = len(rate)
        alive = 0
        dead = 0
        for i in rate:
            if i == 0: dead += 1
            if i == 1: alive += 1
        #print(alive, "/", total)
        #print(dead, "/", total)
        return rate, labels

survived, passenger_id = custom_net()

with open('prediction.txt', 'w') as my_file:
    my_file.write('PassengerId,Survived\n')
    for p, s in zip(passenger_id, survived):
        my_file.write(str(p[0]) + ',' + str(s))
        my_file.write('\n')


