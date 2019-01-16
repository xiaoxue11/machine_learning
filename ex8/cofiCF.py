# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 20:19:24 2019

@author: 29132
"""
import numpy as np
def cofiCostFunc(Theta,X, Y, R, num_users, num_movies, num_features, lamda):
    Theta =Theta.reshape(num_users, num_features)
    J = 0;
    X_grad =np.zeros(X.shape);
    Theta_grad = np.zeros(Theta.shape)
    J1=1/2*sum(sum(np.multiply((np.dot(X,Theta.T)-Y)**2,R)))
    for i in range(num_movies):
        idx=np.where(R[i,:]==1)
        Thetatemp = Theta[idx]
        Ytemp=Y[i,idx]
        X_grad[i,:] = np.dot((np.dot(X[i,:],Thetatemp.T)-Ytemp),Thetatemp)+lamda*X[i,:]
    R1=R.T;
    Y1=Y.T;
    for j in range(num_users):
        idx=np.where(R1[j,:]==1)
        Xtemp = X[idx]
        Ytemp=Y1[j,idx];
        Theta_grad[j,:] = np.dot((np.dot(Theta[j,:],Xtemp.T)-Ytemp),Xtemp)+lamda*Theta[j,:];
    X_grad=X_grad.reshape(np.size(X_grad))
    Theta_grad=Theta_grad.reshape(np.size(Theta_grad))
    grad =Theta_grad
    J=J1+lamda/2*sum(sum(Theta**2))+lamda/2*sum(sum(X**2));
    return J,grad
