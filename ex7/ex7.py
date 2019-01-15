# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 11:13:11 2019

@author: 29132
"""
import scipy.io as scio
import numpy as np
from Findclosedata import findClosestCentroids
from calcenter import computeCentroids
from runkmeans import runkMeans
import matplotlib.image as mpimg 
from kMeansInit import kMeansInitCentroids
import matplotlib.pyplot as plt 
#================= Part 1: Find Closest Centroids ====================
print('Finding closest centroids.')
data1=scio.loadmat('ex7data2.mat')
X=data1['X']
#Select an initial set of centroids
K = 3;
initial_centroids =np.array([[3,3],[6,2],[8,5]]);
idx = findClosestCentroids(X, initial_centroids);
print('Closest centroids for the first 3 examples:',idx[:3])

#===================== Part 2: Compute Means =========================
centroids=computeCentroids(X, idx, K)
print('Centroids computed after initial finding of closest centroids:',centroids)
    
#=================== Part 3: K-Means Clustering ======================
data2=scio.loadmat('ex7data2.mat')
X=data2['X']
K = 3;
max_iters = 10;
initial_centroids =np.array([[3,3],[6,2],[8,5]])
#Run K-Means
[centroids, idx] = runkMeans(X, initial_centroids, max_iters,True);

#============= Part 4: K-Means Clustering on Pixels ===============
A = mpimg.imread('bird_small.png')
K = 16; 
max_iters = 10;
img_size=A.shape
X =A.reshape(img_size[0] * img_size[1], 3);
initial_centroids=kMeansInitCentroids(X,K)
[centroids, idx] = runkMeans(X, initial_centroids, max_iters,False)

#================= Part 5: Image Compression ======================
idx = findClosestCentroids(X, centroids);
X_recovered=np.zeros(X.shape)
for i in range(len(idx)):
    X_recovered[i]= centroids[int(idx[i])]
X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3);
#Display the original image 
plt.figure(1)
plt.subplot(1, 2, 1);
plt.imshow(A); 
plt.title('Original');
plt.subplot(1, 2, 2);
plt.imshow(X_recovered)
plt.title('Compressed, with {} colors.'.format(K))
