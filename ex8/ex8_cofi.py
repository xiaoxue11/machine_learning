# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 16:17:30 2019

@author: 29132
"""

import scipy.io as scio
import numpy as np
#import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
from cofiCF import cofiCostFunc
from loadMovie import loadMovieList
from normRating import normalizeRatings
#=============== Part 1: Loading movie ratings dataset ================
data1=scio.loadmat('ex8_movies.mat');
R=data1['R']
Y=data1['Y']
idx=np.where(R[0,:]==1)
print('Average rating for movie 1 (Toy Story): {}/5'.format(np.mean(Y[0, idx])));
#We can "visualize" the ratings matrix by plotting it with imagesc
plt.imshow(Y);
plt.ylabel('Movies');
plt.xlabel('Users');

#============ Part 2: Collaborative Filtering Cost Function ===========
data2=scio.loadmat('ex8_movieParams.mat')
Theta=data2['Theta']
X=data2['X']
#num_features=data2['num_features']
#num_movies=data2['num_movies']
#num_users=data2['num_users']
num_users = 4; 
num_movies = 5; 
num_features = 3;
X = X[0:num_movies, 0:num_features]
Theta = Theta[0:num_users, 0:num_features]
Y = Y[0:num_movies, 0:num_users]
R = R[0:num_movies, 0:num_users]
Theta=Theta.reshape(np.size(Theta),1)
lamda=0
J,grad=cofiCostFunc(Theta,X, Y, R, num_users, num_movies, num_features, lamda)
print('Cost at loaded parameters: {}'.format(J))

#========= Part 4: Collaborative Filtering Cost Regularization ========
J,grad = cofiCostFunc(Theta, X,Y, R, num_users, num_movies, num_features, lamda=1.5)
print('Cost at loaded parameters (lambda = 1.5):{}'.format(J));

#============== Part 6: Entering ratings for a new user ===============
movieList = loadMovieList()
my_ratings = np.zeros([1682,1])
my_ratings[0] = 4;
my_ratings[97] = 2;
my_ratings[6] = 3;
my_ratings[11]= 5;
my_ratings[53] = 4;
my_ratings[63]= 5;
my_ratings[65]= 3;
my_ratings[68] = 5;
my_ratings[182] = 4;
my_ratings[225] = 5;
my_ratings[354]= 5;
for i in range(len(my_ratings)):
    if my_ratings[i] > 0: 
        print('Rated {}for {}'.format(my_ratings[i], movieList[i+1]));
    
#================== Part 7: Learning Movie Ratings ====================
data3=scio.loadmat('ex8_movies.mat')
R=data3['R']
Y=data3['Y']
Y=np.c_[my_ratings,Y]
my_rating=np.zeros(my_ratings.shape)
for i in range(my_ratings.shape[0]):
    if my_ratings[i]!=0:
        my_rating[i]=1
R=np.c_[my_rating,R]
[Ynorm, Ymean] = normalizeRatings(Y, R)
num_users = Y.shape[1];
num_movies = Y.shape[0];
num_features = 10;
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)
init_theta=Theta.reshape(np.size(Theta))
lamda = 10;
from scipy.optimize import fmin_cg
def cost_func(t):
    return cofiCostFunc(t, X,Ynorm, R, num_users, num_movies, num_features, lamda)[0]
def grad_func(t):
    return cofiCostFunc(t, X,Ynorm, R, num_users, num_movies, num_features, lamda)[1]
theta=fmin_cg(f=cost_func,x0=init_theta,fprime=grad_func)
Theta=theta.reshape(num_users,num_features)

#================== Part 8: Recommendation for you ====================
p = np.dot(X,Theta.T)
my_predictions = p[:,0] + Ymean
movieList = loadMovieList()
idx=np.argsort(-my_predictions)
for i in range(10):
    j = idx[i];
    print('Predicting rating {} for movie {}'.format(my_predictions[j], movieList[j]))
for i in range(len(my_ratings)):
    if my_ratings[i] > 0: 
        print('Rated {} for {}'.format(my_ratings[i],movieList[i+1]))


