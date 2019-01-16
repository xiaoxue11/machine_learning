# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 13:44:08 2019

@author: 29132
"""
import numpy as np
from gaussian import multivariateGaussian
import matplotlib.pyplot as plt
def visualizeFit(X, mu, sigma2):
    x=np.arange(0,35,0.5)
    y=np.arange(0,35,0.5)
    xx,yy =np.meshgrid(x,y);
    X1=np.c_[xx.flatten(),yy.flatten()]
    Z = multivariateGaussian(X1,mu,sigma2);
    Z=Z.reshape(xx.shape)
    plt.plot(X[:, 0], X[:, 1],'bx')
    if (sum(sum(np.isinf(Z)) == 0)):
        levels=10 ** np.arange(-20, 0, 3).astype(np.float)
        plt.contour(xx,yy,Z,levels=levels);
    plt.xlabel('Latency (ms)');
    plt.ylabel('Throughput (mb/s)');