# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:26:39 2019

@author: 29132
"""
import numpy as np
def selectThreshold(pval,yval):
    bestEpsilon = 0;
    bestF1 = 0;
    F1 = 0;
    stepsize = (max(pval) - min(pval)) / 1000;
    epsilon=np.arange(min(pval),max(pval),stepsize)
    for i in range(len(epsilon)):
        m=pval.shape[0]
        predictions=np.zeros(m)
        fp1=np.zeros(m)
        tp1=np.zeros(m)
        fn1=np.zeros(m)
        for j in range(m):
            if (pval[j] < epsilon[i]):
                predictions[j]=1
        for j in range(m):
            if (predictions[j]==1) and (yval[j]==0):
                fp1[j]=1
            if (predictions[j]==1) and (yval[j]==1):
                tp1[j]=1 
            if (predictions[j]==0) and (yval[j]==1):
                fn1[j]=1
        fp=sum(fp1);
        tp=sum(tp1);
        fn=sum(fn1);
        prec=tp/(tp+fp);
        rec=tp/(tp+fn);
        F1=(2*prec*rec)/(prec+rec);
        if F1 > bestF1:
            bestF1 = F1;
            bestEpsilon = epsilon[i];
    return bestEpsilon,bestF1