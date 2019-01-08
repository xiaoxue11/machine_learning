# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 17:18:54 2019

@author: 29132
"""
import numpy as np
import scipy.io as scio
from functions import displayData
from nnCF import nnCostFunction,sigmoidGradient
from InitiaWeights import randInitializeWeights
from checkNNGrad import checkNNGradients
from scipy import optimize
from predict import predict

#Setup the parameters you will use for this exercise
input_layer_size  = 400;  
hidden_layer_size = 25;   
num_labels = 10;         

# =========== Part 1: Loading and Visualizing Data =============
#Load Training Data
print('Loading and Visualizing Data ...')
data=scio.loadmat('ex4data1.mat');
X=data['X']
y=data['y']
[m,n]=np.shape(X)
for i in range(m):
    y[i]-=1
#Randomly select: 100 data points to display
rand_indices=np.arange(m);
np.random.shuffle(rand_indices)
sel = X[rand_indices[0:100], :];
displayData(sel);

#================ Part 2: Loading Parameters ===================
print('Loading Saved Neural Network Parameters ...')
#Load the weights into variables Theta1 and Theta2
data_weight=scio.loadmat('ex4weights.mat');
Theta1=data_weight['Theta1']
Theta2=data_weight['Theta2']
nn_params=np.concatenate([Theta1.reshape(np.size(Theta1)),Theta2.reshape(np.size(Theta2))],axis=0)

#================ Part 3: Compute Cost (Feedforward) ================
print('Feedforward Using Neural Network ...')
# Weight regularization parameter (we set this to 0 here).
lamda = 0;
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)[0]
print('Cost at parameters (loaded from ex4weights): ', J)

#=============== Part 4: Implement Regularization ===============
lamda = 1;
J_reg = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)[0]
print('Cost at parameters with regular (loaded from ex4weights): ', J_reg)

#================ Part 5: Sigmoid Gradient  ================
print('Evaluating sigmoid gradient...')
g = sigmoidGradient(np.array([-1,-0.5,0, 0.5, 1]));
print('Sigmoid gradient evaluated at:',g);

#================ Part 6: Initializing Pameters ================
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params=np.concatenate([initial_Theta1.reshape(np.size(initial_Theta1)),initial_Theta2.reshape(np.size(initial_Theta2))],axis=0)
#=============== Part 7: Implement Backpropagation ===============
print('Checking Backpropagation... ');
#Check gradients by running checkNNGradients
checkNNGradients(0);


#=============== Part 8: Implement Regularization ===============
lamda = 3;
checkNNGradients(lamda)
debug_J  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)[0]
print('Cost at (fixed) debugging parameters',lamda,debug_J)

##=================== Part 9: Training NN ===================
print('Training Neural Network...')
lamda=3
def cost_func(t):
    return nnCostFunction(t, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)[0]
def grad_func(t):
    return nnCostFunction(t, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)[1]
nn_params=optimize.fmin_cg(f=cost_func,x0=initial_nn_params,fprime=grad_func,maxiter=50)
#grad=nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)[1]
Theta1 =nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, (input_layer_size + 1));
Theta2 =nn_params[hidden_layer_size * (input_layer_size + 1):np.size(nn_params)].reshape(num_labels, (hidden_layer_size + 1));

#========================== Part 10: Implement Predict =================
pred = predict(Theta1, Theta2, X,y)
print('The accuracy is: ',pred*100)


