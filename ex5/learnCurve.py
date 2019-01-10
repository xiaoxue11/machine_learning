# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 21:34:46 2019

@author: 29132
"""
import numpy as np
from TLR import trainLinearReg

def learningCurve(X,y,Xval,yval,lamda):
    m=np.shape(X)[0]
    error_train=np.zeros(m)
    error_val=np.zeros(m)
    for i in range(m):
        theta=trainLinearReg(X[:i+1],y[:i+1],lamda);
        m1=X[:i+1].shape[0];
        m2=Xval.shape[0];
        error_train[i]=1/(2*m1)*sum(np.power(np.dot(X[:i+1],theta)-y[:i+1],2));
        error_val[i]=1/(2*m2)*sum(np.power(np.dot(Xval,theta)-yval,2));
    return error_train,error_val
  
    