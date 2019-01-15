# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 20:37:00 2019

@author: 29132
"""
import numpy as np
def projectData(X_norm, U, K):
    Z = np.zeros([X_norm.shape[0], K])
    Ureduce=U[:,0:K];
    Z=np.dot(X_norm,Ureduce)
    return Z
def recoverData(Z,U,K):
    X_rec =np.zeros([Z.shape[0], U.shape[0]])
    Ureduce=U[:,0:K];
    X_rec=np.dot(Z,Ureduce.T)
    return X_rec