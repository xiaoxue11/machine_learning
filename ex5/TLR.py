# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 17:50:33 2019

@author: 29132
"""
import numpy as np
import scipy.optimize as opt
from LinerRegCF import linearRegCostFunction
def trainLinearReg(X, y, lamda):
    y=y.reshape(np.size(y))
    initial_theta=np.zeros(X.shape[1])
    def cost_func(t):
        return linearRegCostFunction(X, y, t, lamda)[0]
    def grad_func(t):
        return linearRegCostFunction(X, y, t, lamda)[1]
    results=opt.fmin_cg(f=cost_func,x0=initial_theta,fprime=grad_func)
    results=results.reshape(np.size(results),1)
    return results