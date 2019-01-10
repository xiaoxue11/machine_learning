# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 23:30:22 2019

@author: 29132
"""
import numpy as np
def polyFeatures(X,p):
    m=X.shape[0]
    X_poly =np.zeros([m, p])
    X=X.reshape(m)
    for i in range(p):
        if i==0:
            X_poly[:,i]=X
        else:
            X_poly[:,i]=np.power(X,i+1)
    return X_poly