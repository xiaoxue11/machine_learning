# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 10:36:20 2019

@author: 29132
"""
import numpy as np
import matplotlib.pyplot as plt
def visualizeBoundary(X,model,h):
    X1=X[:,0]
    X2=X[:,1]
    xmax=max(X1)
    ymax=max(X2)
    xmin=min(X1)
    ymin=min(X2)
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h),np.arange(ymin, ymax, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z)
    return xx,yy,Z
    
    