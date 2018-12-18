# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 17:07:44 2018

@author: 29132
"""
import numpy as np
from computeCost import cost 
def gradientDescent(x, y, theta, alpha, num_iters):
    m=np.size(y)
    J_history=np.zeros((num_iters,1))
    for iters in range(num_iters):
        theta=theta-alpha/m*np.dot(x.T,(np.dot(x,theta)-y))
        J_history[iters]=cost(x,y,theta)
    return theta,J_history
    
    