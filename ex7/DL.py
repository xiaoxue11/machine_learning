# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 17:12:38 2019

@author: 29132
"""
import numpy as np
import matplotlib.pyplot as plt
def drawLine(p1,p2):
    plt.plot(np.array([p1[0],p2[0]]), np.array([p1[1],p2[1]]))