# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 22:07:44 2018

@author: 29132
"""
import numpy as np
import matplotlib.pyplot as plt

def displayData(x):
    (m,n)=x.shape
    example_width=np.round(np.sqrt(n)).astype(int)
    example_height = (n / example_width).astype(int);
#Compute number of items to display
    display_rows = np.floor(np.sqrt(m)).astype(int);
    display_cols = np.ceil(m / display_rows).astype(int);
#Between images padding
    pad = 1;
#Setup blank display
    display_array = -np.ones((pad + display_rows * (example_height + pad), 
                             pad + display_cols * (example_width + pad)));
#Copy each example into a patch on the display array
    curr_ex = 0;
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m :
                break
            max_val = np.max(np.abs(x[curr_ex, :]))
            display_array[pad + (j - 1) * (example_height + pad) + np.arange(example_height),
                    (i - 1) * (example_width + pad) + np.arange(example_width)[:, np.newaxis]] =\
                          x[curr_ex].reshape(example_height, example_width) / max_val;
            curr_ex += 1
        if curr_ex > m:break; 
#Display Image
    plt.figure()
    plt.imshow(display_array, cmap='gray', extent=[-1, 1, -1, 1])
    plt.axis('off')






    