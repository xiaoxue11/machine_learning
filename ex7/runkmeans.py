# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 20:27:36 2019

@author: 29132
"""
import numpy as np
import matplotlib.pyplot as plt
from Findclosedata import findClosestCentroids
from calcenter import computeCentroids
from plotkMeans import plotProgresskMeans
def runkMeans(X, initial_centroids,max_iters,plot_progress):
    [m,n] = X.shape;
    K = initial_centroids.shape[0]
    centroids = initial_centroids;
    previous_centroids = centroids;
    idx = np.zeros(m)
    max_iters=10
    #Run K-Means
    if plot_progress:
        plt.figure(0)
    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids);
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids;
        centroids = computeCentroids(X, idx, K)
    return centroids,idx
    