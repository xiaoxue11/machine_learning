# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 21:56:48 2018

@author: 29132
"""
import numpy as np
def computeCostMulti(x,y,theta,alpha,num_iters):
    m=np.size(y)
    J_history=np.zeros((num_iters,1))
    for i in range(num_iters):
        