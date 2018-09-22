# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.autograd import Variable

class RBM():
    
    def __init__ (self, nv, nh):
#        weight: the probability of all hidden nodes
#        initialize all the weight, which is normalized. weight matrix of p_v_given_h 
        self.W = torch.randn(nh, nv)
#        bias for hidden nodes, which should be 2 dimentional. the first param 1 represents the batch size.
        self.a = torch.randn(1, nh)
#        bias for visible node, which should be 2 dimentionals
        self.b = torch.randn(1, nv)

    def sample_hiddenNodes(self, x):
        '''
        x: verctor of visible neurons (nv)
        '''
#        weight (self.W) * nv, product of the tow tensors
#        print("x:{}, self.W.t():{}".format(x, self.W.t()))
#        print("x.shape:{}, self.W.t().shape:{}".format(x.shape, self.W.t().shape))
        wx = torch.mm(x, self.W.t())
#        make sure the bias are applied to each line of mini-batch
        activation = wx + self.a.expand_as(wx)
#        probability of the activated hidden node give visible node
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
        
    def sample_visibleNodes(self, y):
        '''
        X: verctor of hidden neurons (nh)
        '''
#        weight (self.W) * nv: product of the tow tensors
        wy = torch.mm(y, self.W)
#        make sure the bias are applied to each line of mini-batch
        activation = wy + self.b.expand_as(wy)
#        probability of the activated hidden node give visible node
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)


#        contrastive divergence = approximation of log likelihood of gredianet
#        minimize the energy in energy function = maximizing the likelihood

    def train(self, v0, vk, ph0, phk):
        '''
        v0: the input vactor containing all the movie ratings by one user 
        vk: visible nodes obtained after k Gibbs samplings
        ph0: the vector of probabilities at the first iteration, the hidden nodes eaual 1 given v0, 
        phk:the probabilities of hidden node after k sampling given the visible node variables vk
        '''
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk) 
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)




