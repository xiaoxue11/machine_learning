# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 22:08:51 2019

@author: 29132
"""
import numpy as np
from nnCF import sigmoid
def predict(Theta1,Theta2,X,y):
    [m,n]=np.shape(X)
    X=np.c_[np.ones(m),X]
    z2=np.dot(X,Theta1.T)
    a2=sigmoid(z2)
    a2=np.c_[np.ones(m),a2]
    z3=np.dot(a2,Theta2.T)
    a3=sigmoid(z3)
    p=a3.argmax(axis=1)
    s=np.zeros(m)
    for i in range(m):
        if p[i]==y[i]:
            s[i]=1
    pre=np.mean(s)
    return pre