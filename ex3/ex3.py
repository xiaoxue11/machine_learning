# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 16:07:20 2018

@author: 29132
"""
import numpy as np
import scipy.io as scio
from DisplayData import displayData
from lcf import lrCostFunction
from oneVsAll import oneVsAll
from function import predictOneVsAll


#Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all
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

#============ Part 2a: Vectorize Logistic Regression ============
#Test case for lrCostFunction
print('Testing lrCostFunction() with regularization');
theta_t=np.array([[-2],[-1],[1],[2]])
X_t=np.arange(1,16)/10
X_t=X_t.reshape(3,5)
X_t =np.c_[np.ones([5,1]),X_t.T];
y_t =np.array([[1],[0],[1],[0],[1]]);
lambda_t = 3;
[J,grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

#============ Part 2b: One-vs-All Training ============
print('\nTraining One-vs-All Logistic Regression...')
lamda = 0.1;
all_theta = oneVsAll(X, y, num_labels, lamda)
#================ Part 3: Predict for One-Vs-All ================
pred = predictOneVsAll(all_theta, X,y);
print('Training Set Accuracy: ', pred);

   
    
    
 



