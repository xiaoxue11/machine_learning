# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 20:26:25 2019

@author: 29132
"""
import matplotlib.pyplot as plt
from DL import drawLine

def plotProgresskMeans(X, centroids, previous_centroids, idx, K, m):
    cnames=['#FF0000', '#00FF00','#87CEFA']
    for i in range(len(idx)):
        plt.scatter(X[i,0],X[i,1],s = 15,marker='o',c='',edgecolor=cnames[int(idx[i])])
    plt.plot(centroids[:,0], centroids[:,1], 'kx', markersize=10, linewidth=3)
    for j in range(centroids.shape[0]):
        drawLine(centroids[j, :], previous_centroids[j,:])
    plt.title('Iteration number {}'.format(m))