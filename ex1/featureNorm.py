# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 17:34:32 2018

@author: 29132
"""

import numpy as np
def featureNormalize(x):
    x_norm=x
    n=np.shape(x)[1]
    mu=np.zeros(n)
    sigma=np.zeros(n)
    mu=np.mean(x,axis=0)
    sigma=np.std(x,axis=0)
    x_norm=np.divide((x-mu),sigma)
    return x_norm,mu,sigma
#A=[[1,2],[3,4]]
#A_norm,mu,sigma=featureNormalize(A)
