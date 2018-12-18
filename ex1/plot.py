# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 23:17:59 2018

@author: 29132
"""

import matplotlib.pyplot as plt
def plot_data(x,y):
    plt.plot(x,y,'rx')
    plt.ylabel('Profit in $10,000s'); #Set the ylabel
    plt.xlabel('Population of City in 10,000s')# Set the xlabel
    plt.xlim(4.0, 24.)# set axis limits
    plt.ylim(-5.0, 25.)
    plt.legend(['Training data'])