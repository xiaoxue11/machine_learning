# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 22:12:02 2019

@author: 29132
"""
import numpy as np
import scipy.io as scio
from DisplayData import displayData
from function import predict


input_layer_size  = 400;  #20x20 Input Images of Digits
num_labels = 10; 
#=========== Part 1: Loading and Visualizing Data =============
print("Loading and Visualizing Data ...")
data=scio.loadmat('ex3data1.mat')
X=data['X']
y=data['y']
m=np.size(y)
for i in range(m):
    y[i]-=1
#Randomly select: 100 data points to display
rand_indices=np.arange(m);
np.random.shuffle(rand_indices)
sel = X[rand_indices[0:100], :];
displayData(sel);
#================ Part 2: Loading Pameters ================
print('Loading Saved Neural Network Parameters ...')
data1=scio.loadmat('ex3weights.mat')
Theta1=data1['Theta1']
Theta2=data1['Theta2']
pred = predict(Theta1, Theta2, X,y);
print('Training Set Accuracy: ', pred)