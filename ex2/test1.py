# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 23:06:32 2018

@author: 29132
"""

import numpy as np
from mapf import mapFeature 

u=np.linspace(-1,1,2)
v=np.linspace(-1,1,2)
theta=[[1],[1],[1],[1],[1],[1]]
z=np.zeros((np.size(u),np.size(v)))
for i in range(np.size(u)):
    for j in range(np.size(v)):
        z=np.multiply(u,v,theta)
print(z)
