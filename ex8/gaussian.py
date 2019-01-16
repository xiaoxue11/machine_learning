# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 10:50:43 2019

@author: 29132
"""
import numpy as np
def estimateGaussian(X):
    [m,n]=X.shape
    mu=np.zeros(n)
    sigma2=np.zeros(n)
    mu=np.mean(X,axis=0)
    error=np.zeros(X.shape)
    for i in range(m):
        error[i,:]=X[i,:]-mu;
    sigma2=1/m*sum(np.power(error,2));
    return mu,sigma2

def multivariateGaussian(X, mu, sigma2):
    k=len(mu)
    if sigma2.ndim==1 or (sigma2==2 and (sigma2.shape[0]==1 or sigma2.shape[1]==1)):
        sigma2=np.diag(sigma2)
    X=X-mu
    p=np.zeros(X.shape[0])
    temp=np.sum(np.multiply(np.dot(X,np.linalg.pinv(sigma2)), X),axis=1)
    p=(2 * np.pi) **(- k / 2) * np.linalg.det(sigma2)**(-0.5) *np.exp(-0.5 * temp);
    return p

    