# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 16:44:25 2019

@author: 29132
"""
import numpy as np
def featureNormalize(x):
    n=np.shape(x)[1]
    mu=np.zeros(n)
    sigma=np.zeros(n)
    mu=np.mean(x,axis=0)
    x_norm=x-mu
    sigma=np.std(x_norm,axis=0,ddof=1)
    x_norm=np.divide(x_norm,sigma)
    return x_norm,mu,sigma