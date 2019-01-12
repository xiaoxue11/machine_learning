# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 22:53:46 2019

@author: 29132
"""
import numpy as np
from sklearn import svm
import scipy.io as scio
def error_predic(x,y):
    m=np.size(y)
    error=np.zeros(m)
    for i in range(m):
        if x[i]!=y[i]:
            error[i]=1
    error_rate=np.mean(error)
    return error_rate
def dataset3Params(X, y, Xval, yval):
    C_vec = np.array([0.01,0.03,0.1,0.3,1,3,10,30]);
    sigma_vec =np.array([0.01,0.03,0.1,0.3,1,3,10,30])
    m1=len(C_vec);
    m2=len(sigma_vec);
    C1=0.01;
    sigma1=0.01;
    gamma=1/(2*sigma1*sigma1)
    model=svm.SVC(kernel='rbf', gamma=gamma,C=C1);
    model.fit(X,y)
    predictions = model.predict(Xval);
    error_min=error_predic(predictions,yval)
    for i in range(m1):
        for j in range(m2):
            C=C_vec[i];
            sigma=sigma_vec[j];
            gamma=1/(2*sigma*sigma)
            model= svm.SVC(kernel='rbf', gamma=gamma,C=C); 
            model.fit(X,y)
            predictions = model.predict(Xval);
            error_value=error_predic(predictions,yval)
            if(error_value<=error_min):
                error_min=error_value;
                C1=C;
                sigma1=sigma;
    C=C1;
    sigma=sigma1
    return C,sigma
#data3=scio.loadmat('ex6data3.mat');
#X=data3['X']
#y=data3['y']
#Xval=data3['Xval']
#yval=data3['yval']
#[C,sigma]=dataset3Params(X, y, Xval, yval)

    

    