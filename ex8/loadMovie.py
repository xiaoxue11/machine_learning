# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 22:53:15 2019

@author: 29132
"""
import re
def loadMovieList():
    movieList = {}
    with open('movie_ids.txt') as f:
        for line in f.readlines():
            str1=''
            (key,value)=re.split(' ',line)[0],str1.join(re.split(' ',line)[1:])
            movieList[int(key)]=value
    return movieList


