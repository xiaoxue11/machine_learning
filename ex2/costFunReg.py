# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 22:39:35 2018

@author: 29132
"""

from costFunction import sigmoid
import numpy as np
def costFunctionReg(theta, X, y, lambd):
    m=np.size(y)
    grad=np.zeros(np.size(theta))
    g1=sigmoid(np.dot(X,theta))
    grad1=1/m*np.dot(X.T,g1-y)
    J1=-1/m*sum(np.multiply(y,np.log(g1))+np.multiply(1-y,np.log(1-g1)))
    J=J1+lambd/(2*m)*(sum(np.power(theta,2))-theta[0]**2)
    grad[0]=grad1[0]
    for i in range(1,np.size(theta)):
        grad[i]=grad1[i]+lambd/m*theta[i]
    return J,grad