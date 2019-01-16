# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 10:40:54 2019

@author: 29132
"""
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from gaussian import estimateGaussian,multivariateGaussian
from VF import visualizeFit
from selectTh import selectThreshold
#================== Part 1: Load Example Dataset  ===================
print('Visualizing example dataset for outlier detection.\n\n');
data1=scio.loadmat('ex8data1.mat');
X=data1['X']
Xval=data1['Xval']
yval=data1['yval']
plt.plot(X[:, 0], X[:, 1], 'bx');
plt.axis([0,30,0,30]);
plt.xlabel('Latency (ms)');
plt.ylabel('Throughput (mb/s)');

#================== Part 2: Estimate the dataset statistics ============
[mu,sigma2] = estimateGaussian(X);
p = multivariateGaussian(X, mu, sigma2)
visualizeFit(X,  mu, sigma2);

#================== Part 3: Find Outliers ===================
pval = multivariateGaussian(Xval, mu, sigma2);
[epsilon,F1]=selectThreshold(pval,yval)
print('Best epsilon found using cross-validation:{}'.format(epsilon));
print('Best F1 on Cross Validation Set: {}'.format(F1));
outliers = np.where(p < epsilon)
plt.scatter(X[outliers,0], X[outliers,1], marker='o', c='',edgecolors='r');

#================== Part 4: Multidimensional Outliers ===================
data2=scio.loadmat('ex8data2.mat')
X=data2['X']
Xval=data2['Xval']
yval=data2['yval']
[mu,sigma2] = estimateGaussian(X);
p = multivariateGaussian(X, mu, sigma2);
pval = multivariateGaussian(Xval, mu, sigma2);
epsilon,F1 = selectThreshold(pval,yval);
print('Best epsilon found using cross-validation:{}'.format(epsilon));
print('Best F1 on Cross Validation Set: {}'.format(F1));
m=p.shape[0]
s=np.zeros(m)
for i in range(m):
    if p[i]<epsilon :
        s[i]=1
print('# Outliers found: {}'.format(sum(s)));        
            