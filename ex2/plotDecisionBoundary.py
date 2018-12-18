# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 22:34:07 2018

@author: 29132
"""
from plotData import plotData
import numpy as np
import matplotlib.pyplot as plt
from mapf import mapFeature
def plotDecisionBoundary(theta,X,y):
    plotData(X[:,1:3], y)
    if X.shape[1] <= 3:
# Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:,1])-2, np.max(X[:,1])+2])
        plot_y=(-1/theta[2])*(theta[1]*plot_x+theta[0])
        plt.plot(plot_x, plot_y)
        plt.legend(labels=['Admitted', 'Not admitted','Decision Boundary'])
        plt.axis([30, 100, 30, 100])
    else:
    # Here is the grid range
       u = np.linspace(-1, 1.5, 50);
       v = np.linspace(-1, 1.5, 50);
       z = np.zeros((np.size(u), np.size(v)))
    # Evaluate z = theta*x over the grid
       for i in range(np.size(u)):
           for j in range(np.size(v)):
               z[i,j] = np.dot(mapFeature(u[i],v[j]),theta);
    z = z.T; # important to transpose z before calling contour

    # Plot z = 0
    # Notice you need to specify the range [0, 0]
    plt.contour(u, v, z,[0,1])
