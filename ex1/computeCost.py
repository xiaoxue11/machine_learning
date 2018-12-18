# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 09:47:10 2018

@author: 29132
"""
import numpy as np
def cost(x,y,theta):
    m=np.size(y)
    J=1/(2*m)*np.sum(np.power(np.dot(x,theta)-y,2))
    return J

    