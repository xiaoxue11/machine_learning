# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 14:39:00 2019

@author: 29132
"""
import numpy as np
def sigmoid(z):
    g=1/(1+np.exp(-z))
    return g
def sigmoidGradient(z):
    g1=sigmoid(z)
    g=np.multiply(g1,(1-g1))
    return g
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda):
    m=np.shape(X)[0]
    X=np.c_[np.ones(m),X]
    Y=np.zeros([m,num_labels])
    for j in range(num_labels):
        for i in range(m):
            if y[i]==j:
                Y[i][j]=1
    Theta1=nn_params[0:hidden_layer_size * (input_layer_size + 1)]
    Theta2=nn_params[hidden_layer_size * (input_layer_size + 1):np.size(nn_params)]
    Theta1=Theta1.reshape(hidden_layer_size,(input_layer_size + 1))
    Theta2=Theta2.reshape(num_labels,hidden_layer_size+1) 
    #==forward propagation===============
    z2=np.dot(X,Theta1.T)
    a2=sigmoid(z2)
    a2=np.c_[np.ones(m),a2]
    z3=np.dot(a2,Theta2.T)
    a3=sigmoid(z3)
    J1=-1/m*sum(sum(np.multiply(Y,np.log(a3))+np.multiply((1-Y),np.log(1-a3))))
    #==regular==========================
    J=J1+lamda/(2*m)*(sum(sum(np.power(Theta1[:,1:],2)))+sum(sum(np.power(Theta2[:,1:],2))))
    #==========back propagation====================
    X=X.T
    Y=Y.T
    Theta1_grad =np.zeros(np.shape(Theta1));
    Theta2_grad =np.zeros(np.shape(Theta2));
    for i in range(m):
        a1=X[:,i]
        a1=a1.reshape(np.size(a1),1)
        z2=np.dot(Theta1,a1)
        a2=sigmoid(z2)
        n=np.shape(a2)[1]
        A2=np.r_[np.ones((1,n)),a2]
        z3=np.dot(Theta2,A2)
        a3=sigmoid(z3)
        sigma3=a3-Y[:,i].reshape(np.size(a3),1)
        sigma2=np.multiply(np.dot(Theta2.T,sigma3)[1:],sigmoidGradient(z2))
        Theta2_grad=Theta2_grad+np.dot(sigma3,A2.T)
        Theta1_grad=Theta1_grad+np.dot(sigma2,a1.T)
    grad1=1/m*Theta1_grad
    grad2=1/m*Theta2_grad
    for i in range(hidden_layer_size):
        for j in range(input_layer_size+1):
            if j==0:
                Theta1_grad[i][j]=grad1[i][j]
            else:
                Theta1_grad[i][j]=grad1[i][j]+lamda/m*Theta1[i][j]
    for i in range(num_labels):
        for j in range(hidden_layer_size+1):
            if j==0:
                Theta2_grad[i][j]=grad2[i][j]
            else:
                Theta2_grad[i][j]=grad2[i][j]+lamda/m*Theta2[i][j]
    grad=np.concatenate([Theta1_grad.reshape(np.size(Theta1_grad)),Theta2_grad.reshape(np.size(Theta2_grad))],axis=0)
    return J,grad
