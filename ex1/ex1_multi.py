# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:39:56 2018

@author: 29132
"""

import numpy as np
import matplotlib.pyplot as plt
from featureNorm import featureNormalize
from computeCost import cost
from gradientDe import gradientDescent

#===================load data from txt file=========================
data=np.loadtxt('ex1data2.txt',delimiter=',')
X=data[:,:2]
y=data[:,2]
m=np.size(y)
y=y.reshape(m,1)

#===============Print out some data points=========================
x=X[:10]
print(x)
print(y[:10])


#% Scale features and set them to zero mean
print('Normalizing Features ...')
[X,mu,sigma] = featureNormalize(X)
b=np.ones(m)
X=np.c_[b,X]

#%% ================ Part 2: Gradient Descent ================
print('Testing the cost function ...')
#% compute and display initial cos
theta=np.zeros((3,1))   #initialize fitting parameters
J = cost(X, y, theta)
print('With theta = [0;0;0]\nCost computed = ', J);

#% further testing of the cost function
J = cost(X, y, [[-1],[1],[2]])
print('\nWith theta = [-1;1;2]\nCost computed = ', J);

print('Running gradient descent ...\n')
#% Choose some alpha value
alpha = 0.01;
num_iters = 400

#% Init Theta and Run Gradient Descent 
theta = np.zeros((3, 1))
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

#% Plot the convergence graph
plt.figure(0)
plt.plot(list(range(num_iters)), J_history, '-b');
plt.xlabel('Number of iterations');
plt.ylabel('Cost J');

#% Display gradient descent's result
print('Theta computed from gradient descent:')
print(theta)




