# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 15:29:29 2019

@author: 29132
"""
import numpy as np
def computeCentroids(X, idx, K):
    [m,n]=X.shape
    centroids=np.zeros([K,n])
    for i in range(K):
        centroids[i]=np.mean(X[np.where(idx==i)],axis=0)
    return centroids