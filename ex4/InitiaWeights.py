# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 15:08:21 2019

@author: 29132
"""
import numpy as np
def randInitializeWeights(L_in, L_out):
    W = np.zeros([L_out, 1 + L_in])
    epsilon_init=0.12;
    W=np.random.random((L_out,1+L_in))*2* epsilon_init - epsilon_init;
    return W

def debugInitializeWeights(fan_out, fan_in):
    W =np.zeros([fan_out, 1 + fan_in]);
    W=np.reshape(np.sin(np.arange(np.size(W))),np.shape(W))/10
    return W