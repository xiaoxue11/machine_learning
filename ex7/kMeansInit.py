# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 11:55:51 2019

@author: 29132
"""
import numpy as np
def kMeansInitCentroids(X, K):
    centroids = np.zeros([K, X.shape[1]])
    randidx=np.arange(X.shape[0])
    np.random.shuffle(randidx);
    centroids = X[randidx[0:K], :]
    return centroids