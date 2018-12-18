# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:27:08 2018

@author: 29132
"""
import numpy as np
from plotData import plotData
import matplotlib.pyplot as plt
from mapf import mapFeature
from costFunReg import costFunctionReg
from plotDecisionBoundary import plotDecisionBoundary

#===================plot test data======================
data=np.loadtxt('ex2data2.txt',delimiter=',')
X=data[:,0:2]
y=data[:,2]
plt.figure(0)
plotData(X,y)
plt.legend(labels=['y=1','y=0'],loc='upper right')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

#=========Regularized logistic regression============================
X1=X[:,0]
X2=X[:,1]
X=mapFeature(X1,X2)
initial_theta=np.zeros((X.shape[1],1))
lambd = 1
y=y.reshape((np.size(y),1))
[cost, grad] = costFunctionReg(initial_theta, X, y, lambd);
print('cost:',cost)#(approx): 0.693
print('The grad is:',grad[:5])#0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115
test_theta = np.ones((X.shape[1],1));
[cost, grad] = costFunctionReg(test_theta, X, y, 10);
print('cost:',cost)#3.16
print('The grad is:',grad[:5])#0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922

#============= Part 2: Regularization and Accuracies =============
import scipy.optimize as opt
initial_theta=np.zeros(X.shape[1])
y=y.reshape((np.size(y)))
lamda=1
def cost_func(t):
    return costFunctionReg(t, X, y,lamda)[0]
def grad_func(t):
    grad=costFunctionReg(t, X, y,lamda)[1]
    grad=grad.reshape(np.size(grad))
    return grad 
result=opt.minimize(fun=cost_func,x0=initial_theta,method='BFGS', 
                    jac=grad_func,options={'disp': True})
# Print theta to screen
print('Cost at theta found by fminunc:',result.fun);
theta=np.reshape(result.x,(X.shape[1],1))
print(theta[:5])
plotDecisionBoundary(theta, X, y)
plt.title('lambda = 1' )
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y = 1', 'y = 0', 'Decision boundary'])