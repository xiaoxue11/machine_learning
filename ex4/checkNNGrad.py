# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 15:19:15 2019

@author: 29132
"""
from numpy import linalg as LA
import numpy as np
from InitiaWeights import debugInitializeWeights
from computeNumGrad import computeNumericalGradient
from nnCF import nnCostFunction

def checkNNGradients(t):
    lamda=t
    input_layer_size = 3;
    hidden_layer_size = 5;
    num_labels = 3;
    m = 5;
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
    X  = debugInitializeWeights(m, input_layer_size - 1);
    y  = (1 + np.mod(np.arange(1,m+1), num_labels));
    nn_params=np.concatenate([Theta1.reshape(np.size(Theta1)),Theta2.reshape(np.size(Theta2))],axis=0)
    def cost_func(t):
        return nnCostFunction(t, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)[0]
    def grad_func(t):
        return nnCostFunction(t, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)[1]
    grad=grad_func(nn_params)
    numgrad= computeNumericalGradient(cost_func,nn_params)
    print(numgrad[0:5],grad[0:5])
    diff = LA.norm(numgrad-grad)/LA.norm(numgrad+grad)
    print(diff)
    return diff
