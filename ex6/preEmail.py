# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 23:58:45 2019

@author: 29132
"""
from getVocList import getVocabList
import re
from nltk.stem.porter import PorterStemmer
def processEmail(email_contents):
    vocablist = getVocabList()
    word_indices = []
    email_contents=email_contents.lower()
    email_contents=re.sub('<[^<>]+>', ' ', email_contents)
    email_contents = re.sub('[0-9]+', 'number', email_contents)
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
    email_contents = re.sub('[$]+', 'dollar', email_contents)
    #========================== Tokenize Email ===========================
    l = 0
    stemmer=PorterStemmer()
    tokens = re.split('[@$/#.-:&*+=\[\]?!(){\},\'\">_<;% ]', email_contents)
    for token in tokens:
        token = re.sub('[^a-zA-Z0-9]', '', token)
        token = stemmer.stem(token)
        if len(token) < 1:
            continue;
        for key,value in vocablist.items():
            str1=token;
            str2=value;
            if(str1==str2):
                word_indices.append(key);
        if (l + len(token) + 1) > 78:
            print('\n');
            l = 0;
        l = l + len(token) + 1;
    return word_indices
        
        
    