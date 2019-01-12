# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 00:01:15 2019

@author: 29132
"""
import numpy as np
def predict_accuracy(p,y):
    m=y.shape[0]
    s=np.zeros(m)
    for i in range(m):
        if p[i]==y[i]:
            s[i]=1
    accur=np.mean(s)*100
    return accur