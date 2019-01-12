# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 23:37:25 2019

@author: 29132
"""
import numpy as np
import scipy.io as scio
from sklearn import svm
from preEmail import processEmail 
from emailpro import emailFeatures
from predict import predict_accuracy
from getVocList import getVocabList
#==================== Part 1: Email Preprocessing ====================
with open('emailSample1.txt') as f:
    file_contents=f.read()
word_indices  = processEmail(file_contents)
print(word_indices)

#==================== Part 2: Feature Extraction ====================
print('Extracting features from sample email (emailSample1.txt)')
features = emailFeatures(word_indices);
print('Length of feature vector:', len(features));
print('Number of non-zero entries: ', sum(features > 0))

#=========== Part 3: Train Linear SVM for Spam Classification ========
data1=scio.loadmat('spamTrain.mat');
X=data1['X']
y=data1['y']
C = 0.1;
model = svm.SVC(C=0.1, kernel='linear')
model.fit(X,y.flatten())
p=model.predict(X)
train_accuracy=predict_accuracy(p,y)
print('Training Accuracy:',train_accuracy)

#=================== Part 4: Test Spam Classification ================
data1=scio.loadmat('spamTest.mat');
Xtest=data1['Xtest']
ytest=data1['ytest']
p=model.predict(Xtest)
Test_accuracy=predict_accuracy(p,ytest)
print('Testing Accuracy:', Test_accuracy)

#================= Part 5: Top Predictors of Spam ====================
vocabList = getVocabList();
indices = np.argsort(model.coef_).flatten()[::-1]
print(indices)
for i in range(15):
    print('{} ({:0.6f})'.format(vocabList[indices[i]], model.coef_.flatten()[indices[i]]))



                    
