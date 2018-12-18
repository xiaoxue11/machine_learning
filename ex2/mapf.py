# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 22:07:06 2018

@author: 29132
"""
import numpy as np

def mapFeature(X1,X2):
    m=np.size(X1)
    out=np.ones((m,1))
    for i in range(1,7):
        for j in range(i+1):
            out=np.c_[out,np.multiply(np.power(X1,i-j),np.power(X2,j))]
    return out
    