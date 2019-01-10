# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:47:56 2019

@author: 29132
"""
import numpy as np
from TLR import trainLinearReg
from LinerRegCF import linearRegCostFunction
def validationCurve(X, y, Xval, yval):
    lamda_vec = np.array([0,0.001,0.003,0.01,0.03,0.1,0.3,1,3.0,10]).T
    error_train = np.zeros(np.shape(lamda_vec));
    error_val =np.zeros(np.shape(lamda_vec));
    m=lamda_vec.shape[0]
    for i in range(m):
        lamda = lamda_vec[i]
        theta=trainLinearReg(X,y,lamda);
        error_train[i]=linearRegCostFunction(X, y, theta, 0)[0];
        error_val[i]=linearRegCostFunction(Xval,yval,theta,0)[0]; 
    return lamda_vec,error_train,error_val