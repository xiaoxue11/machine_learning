# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 21:06:36 2019

@author: 29132
"""
import numpy as np
def sigmoid(z):
    g=1/(1+np.exp(-z))
    return g
def predictOneVsAll(all_theta, X,y):
    [m,n]=np.shape(X)
    num_labels=np.shape(all_theta)[0]
    p = np.zeros([m,1])
    prediction=np.zeros([m,num_labels])
    X = np.c_[np.ones([m, 1]),X]
    prediction=np.dot(X,all_theta.T)
    p=prediction.argmax(axis=1)
    s=np.zeros(m)
    for i in range(m):
        if p[i]==y[i]:
            s[i]=1
    pre=np.mean(s)
    return pre
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
    
    
    