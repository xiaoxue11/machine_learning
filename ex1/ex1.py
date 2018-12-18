# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:55:09 2018

@author: 29132
"""
import numpy as np
import warmUp
from plot import plot_data
from computeCost import cost
from gradientDe import gradientDescent
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
## ==================== Part 1: Basic Function ====================
#Complete warmUpExercise.m
print('Running warmUpExercise ... ')
print('5x5 Identity Matrix: ')
warmUp.warmUpExercise()

#======================= Part 2: Plotting =======================
print('Plotting Data ...')
data=np.loadtxt('ex1data1.txt',delimiter=',')
X=data[:,0]
y=data[:,1]
plt.figure(0)
plot_data(X,y)


#=================== Part 3: Cost and Gradient descent ===================
m=np.size(y)
X =np.column_stack((np.ones(m),X)) # % Add a column of ones to x
theta =np.zeros((2, 1))# % initialize fitting parameters
y=y.reshape(m,1)

#% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

print('Testing the cost function ...')
#% compute and display initial cost
J = cost(X, y, theta)
print('With theta = [0 ; 0], Cost computed = {:.2f}'.format(J)) #(approx) 32.07\n
#% further testing of the cost function
J = cost(X, y, [[-1],[2]])
print('\nWith theta = [-1 ; 2], Cost computed = {:.2f}'.format(J));#(approx) 54.24\n');

print('Running Gradient Descent ...')
#% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations)[0];
#% print theta to screen
print('Theta found by gradient descent:',theta);#(approx):-3.6303  1.1664
# Plot the linear fit
plt.plot(X[:,1], np.dot(X,theta), '-')
plt.legend(['Training data','Linear regression'])

#%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...')
#% Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100);

#% initialize J_vals to a matrix of 0's
J_vals = np.zeros((np.size(theta0_vals), np.size(theta1_vals)))

#% Fill out J_vals
for i in range(np.size(theta0_vals)):
    for j in range(np.size(theta1_vals)):
        t =np.row_stack((theta0_vals[i],theta1_vals[j]))
        J_vals[i,j] = cost(X, y, t);

#% Because of the way meshgrids work in the surf command, we need to
#% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T
fig = plt.figure(1)
ax = Axes3D(fig)
theta0_vals,theta1_vals = np.meshgrid(theta0_vals,theta1_vals)
ax.plot_surface(theta0_vals, theta1_vals, J_vals,rstride=1, cstride=1)
plt.xlabel('theta_0'); 
plt.ylabel('theta_1');

#% Contour plot
plt.figure(2);
#% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
plt.xlabel('theta_0'); 
plt.ylabel('theta_1');
#hold on;
plt.plot(theta[0], theta[1], 'rx');