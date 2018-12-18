# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:06:50 2018

@author: 29132
"""

import matplotlib.pyplot as plt
import numpy as np
def plotData(x, y):
    pos=np.where(y==1)
    plt.plot(x[pos][:,0],x[pos][:,1],'k+')
    neg=np.where(y==0)
    plt.plot(x[neg][:,0],x[neg][:,1], 'yo') 

    
    
    
    
    
    