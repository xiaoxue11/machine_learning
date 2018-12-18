# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 15:19:07 2018

@author: 29132
"""
import numpy as np
def sigmoid(x):
    g=1/(1+np.exp(-x))
    return g

def costFunction(test_theta, x, y):
    m=np.size(y)
    grad=np.zeros(np.size(test_theta))
    z=np.dot(x,test_theta)
    prediction=sigmoid(z)
    grad=1/m*np.dot(x.T,prediction-y)
    J=-1/m*sum(np.multiply(y,np.log(prediction))+np.multiply(1-y,np.log(1-prediction)))
    return J,grad
    


    