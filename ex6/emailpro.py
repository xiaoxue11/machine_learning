# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 23:37:22 2019

@author: 29132
"""
import numpy as np
def emailFeatures(word_indices):
    #Total number of words in the dictionary
    n = 1899;
    #You need to return the following variables correctly.
    x=np.zeros([n,1])
    for i in range(len(word_indices)):
        j=word_indices[i];
        x[j]=1;
    return x
        