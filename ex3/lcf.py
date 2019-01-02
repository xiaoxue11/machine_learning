# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 23:59:32 2018

@author: 29132
"""
import numpy as np
def sigmoid(z):
    g=1/(1+np.exp(-z))
    return g

def lrCostFunction(theta,x, y, lambda_t):
    m=np.size(y)
    g=sigmoid(np.dot(x,theta))
    grad=np.zeros(np.shape(theta))
    J1=-1/m*(sum(np.multiply(y,np.log(g)))+sum(np.multiply(1-y,np.log(1-g))))
    J=J1+lambda_t/(2*m)*sum(np.power(theta[1:],2))
    grad1=1/m*(np.dot(x.T,g-y))
    temp=theta
    temp[0]=0
    grad=grad1+lambda_t/m*temp
    return J,grad