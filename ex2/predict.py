# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 22:48:25 2018

@author: 29132
"""
import numpy as np
from costFunction import sigmoid
def predict(theta,X,y):
    m = np.shape(X)[0]
    p = np.zeros((m, 1));
    z=np.dot(X,theta);
    Y=sigmoid(z);
    for i in range(m):
        if Y[i]>=0.5:
            p[i]=1;
        else:
            p[i]=0;
    s=np.zeros(m)
    for i in range(m):
        if p[i]==y[i]:
            s[i]=1
    return s

