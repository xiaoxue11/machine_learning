# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:02:21 2019

@author: 29132
"""
import matplotlib.pyplot as plt
import numpy as np
from ployfeature import polyFeatures
def plotFit(min_x, max_x, mu, sigma, theta, p):
    x1=min_x - 15
    x2=max_x + 25
    x =np.linspace(x1,x2,int((x2-x1)/0.05));
    m=x.shape[0]
    x_poly = polyFeatures(x, p);
    x_poly = x_poly-mu;
    x_poly = x_poly/sigma;
    x_poly = np.c_[np.ones(m), x_poly];
    plt.plot(x, np.dot(x_poly,theta), '--', LineWidth=2)
    plt.show()
    