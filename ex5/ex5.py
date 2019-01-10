# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 15:27:44 2019

@author: 29132
"""
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from LinerRegCF import linearRegCostFunction
from TLR import trainLinearReg
from learnCurve import learningCurve
from ployfeature import polyFeatures
from featureNorm import featureNormalize
from plotfit import plotFit
from validCurve import validationCurve
#=========== Part 1: Loading and Visualizing Data =============
print('Loading and Visualizing Data')
data=scio.loadmat('ex5data1')
X=data['X']
y=data['y']
Xval=data['Xval']
Xtest=data['Xtest']
ytest=data['ytest']
yval=data['yval']
# Plot training data
plt.figure(0)
plt.plot(X, y, 'rx', MarkerSize=10, LineWidth=1.5);
plt.xlabel('Change in water level (x)');
plt.ylabel('Water flowing out of the dam (y)');

# =========== Part 2: Regularized Linear Regression Cost =============
theta = np.array([[1.0],[1.0]]);
lamda=1
m=np.size(y)
[J,grad] = linearRegCostFunction(np.c_[np.ones(m),X], y, theta, lamda);
print('Cost', J);

#=========== Part 3: Regularized Linear Regression Gradient =============
[J,grad] = linearRegCostFunction(np.c_[np.ones(m),X], y, theta, lamda);
print('Gradient: ', grad);

#=========== Part 4: Train Linear Regression =============
lamda=0
theta=trainLinearReg(np.c_[np.ones(m),X], y, lamda)
#Plot fit over the data
plt.plot(X, np.dot(np.c_[np.ones(m),X],theta), '--')
plt.xlabel('Change in water level (x)');
plt.ylabel('Water flowing out of the dam (y)');

#=========== Part 5: Learning Curve for Linear Regression =============
lamda=0
[error_train, error_val] = learningCurve(np.c_[np.ones(m),X], y,np.c_[np.ones(np.size(Xval)),Xval], yval, lamda);
#Plot fit over the data
plt.figure(1)
plt.plot(range(m), error_train)
plt.plot(range(m), error_val)
plt.title('Learning curve for linear regression')
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('Number of training examples')
plt.ylabel('Error')
print(error_train,error_val)

#=========== Part 6: Feature Mapping for Polynomial Regression =============
p = 8;
# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly)
X_poly = np.c_[np.ones(m), X_poly]
# Map X_poly_test and normalize (using mu and sigma)
mtest=Xtest.shape[0]
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = X_poly_test-mu;
X_poly_test =X_poly_test/sigma
X_poly_test = np.c_[np.ones(mtest), X_poly_test]   
# Map X_poly_val and normalize (using mu and sigma)
mval=Xval.shape[0]
X_poly_val = polyFeatures(Xval, p);
X_poly_val = X_poly_val-mu;
X_poly_val = X_poly_val/sigma;
X_poly_val = np.c_[np.ones(mval), X_poly_val]; 

##=========== Part 7: Learning Curve for Polynomial Regression =============
lamda = 0;
theta = trainLinearReg(X_poly, y, lamda);
#Plot training data and fit
plt.figure(2);
plt.plot(X, y, 'rx', MarkerSize=10, LineWidth=1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);
plt.xlabel('Change in water level (x)');
plt.ylabel('Water flowing out of the dam (y)');
plt.title ('Polynomial Regression Fit (lamda = {})'.format(lamda));
plt.figure(3);
[error_train, error_val] =learningCurve(X_poly, y, X_poly_val, yval, lamda);
plt.plot(range(m), error_train)
plt.plot(range(m), error_val);
plt.title('Polynomial Regression Learning Curve (lamda = {})'.format(lamda));
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0,13,0,100])
plt.legend(['Train', 'Cross Validation'])
print('Polynomial Regression ', lamda);
print(error_train,error_val);

#=========== Part 8: Validation for Selecting Lambda =============
[lambda_vec, error_train, error_val] = validationCurve(X_poly, y, X_poly_val, yval)
plt.figure(4)
plt.plot(lambda_vec, error_train)
plt.plot(lambda_vec, error_val)
plt.legend(['Train', 'Cross Validation']);
plt.xlabel('lambda');
plt.ylabel('Error');
print(error_train)
print(error_val)
