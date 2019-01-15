# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 11:35:52 2019

@author: 29132
"""
import numpy as np
def findClosestCentroids(X,centroids):
    K=centroids.shape[0]
    idx=np.zeros(X.shape[0])
    c=np.zeros(K)
    for i in range(X.shape[0]):
        for j in range(K):
            c[j]=sum(np.power(X[i,:]-centroids[j,:],2))
        index=c.argmin(axis=0)
        idx[i]=index
    return idx