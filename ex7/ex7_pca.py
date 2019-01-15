# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 16:22:37 2019

@author: 29132
"""

import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt 
from featureNorm import featureNormalize
from PCA import pca
from DL import drawLine
from prdata import projectData,recoverData
from Displaydata import displayData
import matplotlib.image as mpimg 
from kMeansInit import kMeansInitCentroids
from runkmeans import runkMeans
# ================== Part 1: Load Example Dataset  ===================
data1=scio.loadmat('ex7data1.mat')
X=data1['X']
plt.figure(0)
plt.plot(X[:,0],X[:,1],'bo')
plt.axis([0.5,6.5,2,8])

# =============== Part 2: Principal Component Analysis ===============
#%  Before running PCA, it is important to first normalize X
[X_norm, mu, sigma] = featureNormalize(X);
[U, S] = pca(X_norm);
drawLine(mu, mu + 1.5 * S[0] * U[:,0].T)
drawLine(mu, mu + 1.5 * S[1] * U[:,1].T)
print('U(:,1) ={},{}'.format(U[0][0], U[1][0]));

#=================== Part 3: Dimension Reduction ===================
plt.figure(1)
plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo');
plt.axis([-4,3,-4,3])
#Project the data onto K = 1 dimension
K = 1;
Z = projectData(X_norm, U, K);
print('Projection of the first example:{}'.format(Z[0]));

X_rec  = recoverData(Z, U, K)
print('Approximation of the first example:{} {}'.format(X_rec[0][0], X_rec[0][1]))
#Draw lines connecting the projected points to the original points
plt.figure()
plt.plot(X_rec[:,0],X[:,1],'ro')
for i in range(X_norm.shape[0]):
    plt.plot(X_norm[i,:], X_rec[i,:],'k-', Linewidth=1)

#=============== Part 4: Loading and Visualizing Face Data =============
data2=scio.loadmat('ex7faces.mat')
X=data2['X']
displayData(X[0:100, :]);

#=========== Part 5: PCA on Face Data: Eigenfaces  ===================
[X_norm, mu, sigma] = featureNormalize(X)
[U, S] = pca(X_norm)
displayData(U[:, 0:36].T)

#============= Part 6: Dimension Reduction for Faces =================
K = 100;
Z = projectData(X_norm, U, K);

#==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
K = 100;
X_rec  = recoverData(Z, U, K);
#Display normalized data
plt.subplot(1, 2, 1);
displayData(X_norm[0:100,:]);
plt.title('Original faces');

#Display reconstructed data from only k eigenfaces
plt.subplot(1, 2, 2);
displayData(X_rec[0:100,:]);
plt.title('Recovered faces');

#=== Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
from mpl_toolkits.mplot3d import Axes3D
A = mpimg.imread('bird_small.png')
K = 16; 
max_iters = 10;
img_size=A.shape
X =A.reshape(img_size[0] * img_size[1], 3);
initial_centroids=kMeansInitCentroids(X,K)
[centroids, idx] = runkMeans(X, initial_centroids, max_iters,False)
randidx=np.arange(1000)
np.random.shuffle(randidx);
fig = plt.figure()
ax = Axes3D(fig)
for i in range(len(randidx)):
    ax.scatter(X[randidx[i]][0],X[randidx[i]][1],X[randidx[i]][2])
plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')

# === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
[X_norm, mu, sigma] = featureNormalize(X)
[U, S] = pca(X_norm);
Z = projectData(X_norm, U, 2);
plt.figure()
for i in range(len(randidx)):
    plt.scatter(Z[randidx[i],0],Z[randidx[i],1])
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')