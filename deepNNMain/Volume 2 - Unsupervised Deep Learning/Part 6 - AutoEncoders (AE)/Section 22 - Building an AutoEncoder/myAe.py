# AutoEncoders

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import time

#import the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header = None, engine = 'python', encoding = 'latin-1' )
users = pd.read_csv('ml-1m/users.dat', sep='::', header = None, engine = 'python', encoding = 'latin-1' )
ratings = pd.read_csv('ml-1m/users.dat', sep='::', header = None, engine = 'python', encoding = 'latin-1' )

#prepare the training and test dataset
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t', header = None )
training_set_int = np.array(training_set, dtype='int') 
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t', header = None )
test_set_int = np.array(test_set, dtype='int') 

#get the number of total users and movies
nb_users = int(max(max(training_set_int[:, 0]), max(test_set_int[:, 0])))
nb_movies = int(max(max(training_set_int[:, 1]), max(test_set_int[:, 1])))

#convert the data into an array with users in lines and movies in columns
def convertForTorchAe(data):
    new_data = []
    for id_users in range (1, nb_users + 1):
#        get the data only having the same user id as id_users 
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
#        only the index not having the ratings at id_ratings get 0
        ratings[id_movies - 1] = id_ratings
#        torch package requires list of list 
        new_data.append(list(ratings))
    return new_data

training_set_converted = convertForTorchAe(training_set_int)
test_set_converted = convertForTorchAe(test_set_int)

#convert the data into torch tensors
training_set_torch = torch.FloatTensor(training_set_converted)
test_set_torch = torch.FloatTensor(test_set_converted)

#create the architecture of auto encoder NN
class StackedAutoEncoder (nn.Module):
    
    def __init__(self, ):
#        call the constructor of the parent class
        super(StackedAutoEncoder, self).__init__()
#        defining the connection between the input layer and the first hidden layer
        self.fullConnection1 = nn.Linear(in_features=nb_movies, out_features=20)
        self.fullConnection2 = nn.Linear(in_features=20, out_features=10)
        self.fullConnection3 = nn.Linear(in_features=10, out_features=20)
        self.fullConnection4 = nn.Linear(in_features=20, out_features=nb_movies)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fullConnection1(x))
        x = self.activation(self.fullConnection2(x))
        x = self.activation(self.fullConnection3(x))
        x = self.fullConnection4(x)
        return x

sae = StackedAutoEncoder()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(params = sae.parameters(), lr=0.01, weight_decay=0.5)
#print('optimizer: ',optimizer)

#train the stacked auto encoder
nb_epoch = 200
#loop for epoch
for epoch in range (1, nb_epoch + 1):
    train_loss = 0
#    the number of users rating at least one movie (exclue the one giving no rating) 
    s = 0.
    
#    loop over each user
    for id_user in range (nb_users):
        input = Variable(training_set_torch[id_user]).unsqueeze(0)
#        print('Variable(training_set_torch[id_user]): ', Variable(training_set_torch[id_user]))
#        print('input: ', input)
        target = input.clone()
#        print('target: ', target)
#        just eliminate the user having no rating
        if torch.sum(target.data > 0) > 0:
            output = sae.forward(input)
#            print('output: ', output)
#            not update target by gradient descent
            target.require_grad = False
#            not calculate the weight of the 0 ratings
            output[target == 0] = 0
#            calculate the lossfunction. Average of the differences.of the real rating and predicted rating
            loss = criterion(output, target)
#            print('loss: ', loss)
#            average of the error only considering the rated movies, not all of the movies
            mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-15)
#            print('mean_corrector: ', mean_corrector)
#            calculate the backward, which decided the direction of updates (plus or minus) 
            loss.backward()
            train_loss += np.sqrt(loss.data[0] * mean_corrector)
            s += 1.
#            apply the optimizer, which decide the intensity (amount) of the backward update
            optimizer.step()
            
#    if loss is 1, the difference between the prediction and the real ratings is 1 star. it is expected to be less than 1 at least.
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))

####################
#test the sae
####################
#we predict the ratings of the movies that users have not watched

test_loss = 0
#    the number of users rating at least one movie (exclue the one giving no rating) 
s = 0.
for id_user in range (nb_users):
    input = Variable(training_set_torch[id_user]).unsqueeze(0)
    target = Variable(test_set_torch[id_user])
#        print('target: ', target)
#        just eliminate the user having no rating
    if torch.sum(target.data > 0) > 0:
        output = sae.forward(input)
#            print('output: ', output)
#            not update target by gradient descent
        target.require_grad = False
#            not calculate the weight of the 0 ratings
        output[target == 0] = 0
#            calculate the lossfunction. Average of the differences.of the real rating and predicted rating
        loss = criterion(output, target)
#            print('loss: ', loss)
#            average of the error only considering the rated movies, not all of the movies
        mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-15)
#            print('mean_corrector: ', mean_corrector)
#            calculate the backward, which decided the direction of updates (plus or minus) 
        test_loss += np.sqrt(loss.data[0] * mean_corrector)
        s += 1.
        
#    if loss is 1, the difference between the prediction and the real ratings is 1 star. it is expected to be less than 1 at least.
print('test loss: ' + str(test_loss / s))

