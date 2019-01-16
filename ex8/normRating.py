# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 23:44:52 2019

@author: 29132
"""
import numpy as np
def normalizeRatings(Y,R):
    [m, n] = Y.shape
    Ymean = np.zeros(m);
    Ynorm = np.zeros(Y.shape);
    for i in range(m):
        idx = np.where(R[i, :] == 1)
        Ymean[i] = np.mean(Y[i, idx]);
        Ynorm[i, idx] = Y[i, idx] - Ymean[i];
    return Ynorm,Ymean