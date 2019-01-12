# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 08:46:30 2019

@author: 29132
"""
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from plotdata import plotData
from sklearn import svm
from VBL import visualizeBoundary
from GKernel import gaussianKernel
from DataParams import dataset3Params

#=============== Part 1: Loading and Visualizing Data ================
print('Loading and Visualizing Data')
#load data
data1=scio.loadmat('ex6data1.mat');
X=data1['X']
y=data1['y']
#plot training data
plt.figure(0)
plotData(X, y);
#==================== Part 2: Training Linear SVM ====================
print('Training Linear SVM')
C = 1;
linear_svc = svm.SVC(C=1,kernel ='linear'); 
linear_svc.fit(X,y)
h=0.02
visualizeBoundary(X,linear_svc,h);

linear_svc = svm.SVC(C=100,kernel ='linear'); 
linear_svc.fit(X,y)
visualizeBoundary(X,linear_svc,h);
#=============== Part 3: Implementing Gaussian Kernel ===============
x1 = np.array([1,2,1]); 
x2 =np.array([0,4,-1]); 
sigma = 2;
sim = gaussianKernel(x1, x2, sigma);
#=============== Part 4: Visualizing Dataset 2 ================
data2=scio.loadmat('ex6data2.mat');
X=data2['X']
y=data2['y']
#plot training data
plt.figure(1)
plotData(X, y);
#========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
C=1
sigma = 0.1
gamma=1/(2*sigma*sigma)
model=svm.SVC(kernel='rbf', gamma=gamma,C=1)
model.fit(X,y)
h=0.02
visualizeBoundary(X,model,h);

#=============== Part 6: Visualizing Dataset 3 ================
data3=scio.loadmat('ex6data3.mat');
X=data3['X']
y=data3['y']
Xval=data3['Xval']
yval=data3['yval']
#plot training data
plt.figure(2)
plotData(X, y);

##========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========
[C, sigma] = dataset3Params(X, y, Xval, yval);
#Train the SVM
gamma=1/(2*sigma*sigma)
model= svm.SVC(C=C,kernel='rbf', gamma=gamma);
model.fit(X,y)
h=0.02
visualizeBoundary(X,model,h);

