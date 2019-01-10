# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 16:15:01 2019

@author: 29132
"""
import numpy as np
def linearRegCostFunction(X, y, theta, lamda):
    m=np.size(y)
    J=1/(2*m)*sum(np.power(np.dot(X,theta)-y,2))+lamda/(2*m)*sum(np.power(theta[1:],2))
    grad=np.zeros(np.shape(theta))
    grad=1/m*np.dot(X.T,np.dot(X,theta)-y)
    for i in range(np.size(theta)):
        if i!=0:
            grad[i]=grad[i]+lamda/m*sum(np.power(theta[1:],2))
    grad=grad.reshape(np.size(grad))
    return J,grad
    