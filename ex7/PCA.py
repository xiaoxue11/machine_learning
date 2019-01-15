# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 16:52:06 2019

@author: 29132
"""
import numpy as np
def pca(x):
    [m, n] = x.shape
    U=np.zeros(n)
    S=np.zeros(n)
    Sigma=1/m*np.dot(x.T,x)
    [U,S,_]=np.linalg.svd(Sigma)
    return U,S
    
    