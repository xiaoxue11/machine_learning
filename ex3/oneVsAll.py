# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 23:55:09 2018

@author: 29132
"""
import numpy as np
from lcf import lrCostFunction
from scipy import optimize
def oneVsAll(X, y, num_labels, lamda):
    [m,n]=np.shape(X)
    all_theta=np.zeros([num_labels,n+1])
    X=np.c_[np.ones([m,1]),X]
    Y=np.zeros([m,num_labels])
    for j in range(num_labels):
        for i in range(m):
            if y[i]==j:
                Y[i][j]=1
    initial_theta = np.zeros(n + 1)
    def cost_fun(t):
        return lrCostFunction(t,X,y,lamda)[0]
    def grad_fun(t):
        return lrCostFunction(t, X, y, lamda)[1]
    for i in range(num_labels):
        y=Y[:,i]
        all_theta[i:,]=optimize.fmin_cg(f=cost_fun,x0=initial_theta,fprime=grad_fun)
    return all_theta
    
    