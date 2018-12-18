# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 17:58:50 2018

@author: 29132
"""
import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData
from costFunction import costFunction
from costFunction import sigmoid
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict

#===================import data============================
data=np.loadtxt('ex2data1.txt',delimiter=',')
X=data[:,:2]
y=data[:,2]
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')
plt.figure(0)
plotData(X,y)
plt.legend(labels = ['Admitted','Not Admitted'], loc = 'upper right')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

### ============ Part 2: Compute Cost and Gradient ============
[m,n]=np.shape(X)
X=np.column_stack((np.ones(m),X))
y=y.reshape(m,1)
initial_theta = np.zeros((n + 1, 1))
# Compute and display initial cost and gradient
[cost,grad] = costFunction(initial_theta, X, y);
print('Cost at initial theta (zeros):', cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): ',grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');

# Compute and display cost and gradient with non-zero theta
test_theta = [[-24],[0.2], [0.2]];
[cost,grad] = costFunction(test_theta, X, y);
#grad = costFunction.Gradient(test_theta, X, y);
print('Cost at test theta: ', cost);#0.218
print('Gradient at test theta:',grad);#0.043\n 2.566\n 2.647\n

## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.
import scipy.optimize as opt
initial_theta=np.zeros(n+1)
y=y.reshape(m)
def cost_func(t):
    return costFunction(t, X, y)[0]
def grad_func(t):
    return costFunction(t, X, y)[1]

# Run fmin_bfgs to obtain the optimal theta
#theta, cost, *unused = opt.fmin_bfgs(f=cost_func, fprime=grad_func, x0=initial_theta,
#                                    maxiter=400, full_output=True, disp=False)
result=opt.minimize(fun=cost_func,x0=initial_theta,method='BFGS', 
                    jac=grad_func,options={'disp': True})
# Print theta to screen
print('Cost at theta found by fminunc:',result.fun);
#print('Expected cost (approx): 0.203\n');
theta=np.reshape(result.x,(n+1,1))
print(theta);
print(' -25.161\n 0.206\n 0.201\n');

# Plot Boundary
plt.figure(1)
plotDecisionBoundary(theta, X, y);
#============== Part 4: Predict and Accuracies ==============
temp=[1,45,85]
prob = sigmoid(np.dot(temp,theta));
print('we predict an admission probability of ',prob);#0.775 +/- 0.002

# Compute accuracy on our training set
s = predict(theta, X, y);
result=np.mean(s)*100
print('Train Accuracy:', result)







