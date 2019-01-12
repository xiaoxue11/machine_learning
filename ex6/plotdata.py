# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 08:57:28 2019

@author: 29132
"""
import numpy as np 
import matplotlib.pyplot as plt
def plotData(X,y):
    m=X.shape[0]
    X1=X[:,0].reshape(m,1)
    X2=X[:,1].reshape(m,1)
    pos=np.where(y==1)
    plt.plot(X1[pos],X2[pos],'k+',LineWidth=1, markersize=7)
    neg=np.where(y==0)
    plt.plot(X1[neg], X2[neg], 'ko', markerfacecolor='y', markersize=7)
    plt.show