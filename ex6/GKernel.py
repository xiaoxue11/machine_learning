# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 22:00:32 2019

@author: 29132
"""
import numpy as np
def gaussianKernel(x1, x2, sigma):
    sim = 0
    sim=np.exp(-sum(np.power((x1-x2),2))/(2*sigma*sigma))
    return sim
    