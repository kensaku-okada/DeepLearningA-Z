# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.autograd import Variable
import myRbmClass

#########################################
#preprocessing start
#########################################
#import the movie dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
#import the users dataset
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
#import the ratings dataset
rating = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

#import the training and test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t', header = None)
#print("training_set ", training_set)
training_set_array = np.array(training_set, dtype = 'int')
#print("training_set_array ", training_set_array)
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t', header = None)
test_set_array = np.array(training_set, dtype = 'int')

#get the number of users and movies
nb_users = int(max(max(training_set_array[:, 0]), max(test_set_array[:, 0])))
nb_movies = int(max(max(training_set_array[:, 1]), max(test_set_array[:, 1])))

#convert the data into an array with users in lines and movies in columns
def convertArrayToList(data):
    new_data = []
    for userId in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == userId]
        id_ratings = data[:, 2][data[:, 0] == userId]
        ratings = np.zeros(nb_movies)
#        print("ratings[id_movies - 1].shape:", ratings[id_movies - 1].shape)
        ratings[id_movies - 1] = id_ratings
#        print("ratings.shape:", ratings.shape)
#        print("ratings[id_movies - 1]:", ratings[id_movies - 1])
#        print("ratings:", ratings)
        new_data.append(list(ratings))
#        print('new_data: ',new_data)
    return new_data

training_set_list = convertArrayToList(training_set_array)
test_set_list = convertArrayToList(test_set_array)

#convert the data into torch tensors (a matrix of a single data type)
training_set_torchTensor = torch.FloatTensor(training_set_list)
#print("training_set_torchTensor: ",training_set_torchTensor)
test_set_torchTensor = torch.FloatTensor(test_set_list)
#print("test_set_torchTensor: ",test_set_torchTensor)

#convert the ratings into binary ratings 1 (liked) or 0 (not liked) for RBM format because the output value of RBM is binary
#no rating 0 -> -1
training_set_torchTensor[training_set_torchTensor == 0] = -1
training_set_torchTensor[training_set_torchTensor == 1] = 0
training_set_torchTensor[training_set_torchTensor == 2] = 0
training_set_torchTensor[training_set_torchTensor >= 3] = 1
test_set_torchTensor[test_set_torchTensor == 0] = -1
test_set_torchTensor[test_set_torchTensor == 1] = 0
test_set_torchTensor[test_set_torchTensor == 2] = 0
test_set_torchTensor[test_set_torchTensor >= 3] = 1
#########################################
#preprocessing end
#########################################

#create the architecture/object of the neural network

#number of movies == nv
nv = len(training_set_torchTensor[0])
#the parameter we choose == nh (number of hidden node) == number of features detected by the RBM model
nh = 100
#the parameter we choose
batchSize = 10
#batchSize = 100

rbm = myRbmClass.RBM(nv, nh)

##############################
#train the RBM
##############################
nb_epoch = 10

for epoch in range(nb_epoch + 1):
#    loss function output
    train_loss = 0
#    counter just for normalizing the train loss
    s = 0.0
    
#    looped for each user
    for id_user in range (0, nb_users - batchSize, batchSize):
#        input vector, which is gonna be updated by RBM
        vk = test_set_torchTensor[id_user : id_user + batchSize]
#        the target vector, which is not gonnabe updated but compared with vk to calculate the loss
        v0 = test_set_torchTensor[id_user : id_user + batchSize]
#        initial probability
        ph0, _ = rbm.sample_hiddenNodes(v0) 
#        loop for k-steps contrastive divergence, (k-steps random walk Gibbs sampling == Marcov chain monte carlo technique)
        for k in range (10):
#            update the hidden node at k th step of contrasitve divergence
            _ , hk = rbm.sample_hiddenNodes(vk)
#            update the sampled visible node after the k th step of Gibbs sampling 
            _ , vk = rbm.sample_visibleNodes(hk)
#            the ratings of the movies that were not rated were all -1. we want to keep these ratings -1 (non rated)
            vk[v0 < 0] = v0[v0 < 0]
            
#        get phk fortraining
        phk, _ = rbm.sample_hiddenNodes(vk)
#        approximate the gradient to update the biases and the hidden node weights towards the direction of maximum likelihood
        rbm.train(v0, vk, ph0, phk)
#        update the train loss
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
#        if you want to calc the loss function with RMSE (Root Mean Square Error), do like this
#        train_loss += np.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2))

        s += 1. 
#    we want to show the normalized train loss
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss/s))

#############################
#test the RBM
#############################

test_loss = 0
s = 0.0

#    looped for each user
for id_user in range (0, nb_users):
#    you must not use test_set_torchTensor here
    v = training_set_torchTensor[id_user : id_user + 1]
#    you must use test_set_torchTensor only here
    vTarget = test_set_torchTensor[id_user : id_user + 1]
#    eliminate the user giving only -1 ratings (not rated movies)
    if len(vTarget[vTarget >= 0]) > 0:
        _ , h = rbm.sample_hiddenNodes(v)
        _ , v = rbm.sample_visibleNodes(h)
    #        update the test loss
        test_loss += torch.mean(torch.abs(vTarget[vTarget >= 0] - v[vTarget >= 0]))
        s += 1. 
#        if you want to calc the loss function with RMSE (Root Mean Square Error), do like this
#        test_loss += np.sqrt(torch.mean((vTarget[vTarget >= 0] - v[vTarget >= 0])**2))
        
        
print('test loss' + str(test_loss/s))


