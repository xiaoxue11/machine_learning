# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 15:13:43 2019

@author: 29132
"""
import numpy as np
def computeNumericalGradient(J, theta):
    m=np.size(theta)
    numgrad =np.zeros(m);
    perturb= np.zeros(m);
    e = 1e-4;
    for i in range(m):
        perturb[i] = e;
        loss1 =J(theta - perturb);
        loss2 =J(theta + perturb);
        numgrad[i]= (loss2 - loss1) / (2*e);
        perturb[i]= 0;
    return numgrad