# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 09:23:39 2019

@author: 29132
"""
def getVocabList():
    vocablist = {}
    with open('vocab.txt') as f:
        for line in f.readlines():
            (key,value) = line.split()
            vocablist[int(key)] = value
    return vocablist
            
        