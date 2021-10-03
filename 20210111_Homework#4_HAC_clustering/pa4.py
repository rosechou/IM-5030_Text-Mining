# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:35:37 2021

@author: user
"""

# install NLTK  
import nltk
#import string
# install related NLTK packages 
nltk.download()
# Porter’s algorithm
from nltk.stem.porter import *
#stopwords package
from nltk.corpus import stopwords
# to read all files in folder

import os
import pandas as pd
import numpy as np

from numpy import dot
from numpy.linalg import norm

import pickle
import tqdm
from tqdm import tqdm
import heapq

# Stemming using Porter’s algorithm
ps = PorterStemmer() 
# Stopword lists
stop_words = set(stopwords.words('english')) #Stopword
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',
                   '\'s','\'m','\'re','\'ll','\'d','n\'t','shan\'t','that\'s',"'at",
         "_","`","\'\'","--","``",".,","//",":","___",'_the','-',"'em",".com","...","\'ve",'u']) 
# print(stop_words)


def preprocessing(texts):
    texts = texts.translate(str.maketrans('', '', "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"))
    # using translate and digits 
    # to remove numeric digits from string 
    # words in dictionary: 15144-->14348
    remove_digits = str.maketrans('', '', "0123456789") 
    texts = texts.translate(remove_digits) 
    
    word_tokenize = texts.split()
    tokens = [i.lower() for i in word_tokenize if i.lower() not in stop_words]  #Stopword removal
    #tokens = [i for i in tokens if not i.isdigit()] 
    #print(tokens)
    token_result = ''
    for i,token in enumerate(tokens):
        if i != len(tokens)-1: 
            token_result += ps.stem(token) + ' '
        else:
            token_result += ps.stem(token)
    return(token_result)

def similarity(doc1, doc2):
    #separate the column with space
    d1 = pd.read_csv(doc1,names=['t_index','tf_idf'], sep=' ') 
    d2 = pd.read_csv(doc2,names=['t_index','tf_idf'], sep=' ')
    #remove the first row(counter)
    d1 = d1.drop(0) 
    d2 = d2.drop(0)
    #d2
    d1_d2 = pd.merge(d1,d2,on='t_index', how='outer')
    d1_d2.fillna(0,inplace=True)
    matrix_product = np.sum(dot(d1_d2.tf_idf_x, d1_d2.tf_idf_y))
    #print(matrix_product)
    #print(norm(d1_d2.tf_idf_x)*norm(d1_d2.tf_idf_y))
    sim = matrix_product / (norm(d1_d2.tf_idf_x)*norm(d1_d2.tf_idf_y))
    return sim

#sim = similarity('C:/Users/user/R08725008/tf_idf/1.txt','C:/Users/user/R08725008/tf_idf/2.txt')
#print(str(sim)) 
    
def merge_sim(C, j, i, m):
    x = np.max([C[j][i], C[j][m]]) #single link
    y = np.min([C[j][i], C[j][m]]) # complete link
    return (x+y)/2


def heap_merge_sim(C, j, i, m):
    x = np.min([C[j][i][0], C[j][m][0]]) #single link
    y = np.max([C[j][i][0], C[j][m][0]]) # complete link
    return (x+y)/2

N = 1095
eps = 1e-10
C = np.zeros([N,N])

for i in tqdm(range(N)):
    for j in range(N-i-1):
        sim = similarity('C:/Users/user/R08725008/tf_idf/'+str(i+1)+'.txt', 'C:/Users/user/R08725008/tf_idf/'+str(j+i+2)+'.txt')
        C[i][j+i+1] = C[j+i+1][i] = sim + eps
print(C.shape)
pickle.dump(obj=C, file=open('C:/Users/user/R08725008/C.pkl','wb'))


result = []
basic_result = []
C = pickle.load(open('C:/Users/user/R08725008/C.pkl','rb'))
I = np.ones((N,), dtype=int)
K = [8, 13, 20]
for n in range(N):
    result.append([n])

# basic version of HAC
A = [] # a record list of merges
for k in tqdm(range(N - 1)):
    max_sim = 0
    max_i = 0
    max_m = 0
    for i in range(N): # argmax<i,m>
        for m in range(i + 1):
            if i != m and I[i] == 1 and I[m] == 1 and C[i][m] >= max_sim:
                max_sim = C[i][m]
                max_i = i
                max_m = m
                
    A.append((max_i, max_m))


    result[max_i] += result[max_m] # merge m in i
    result[max_m] = None
    
    for j in range(N): #update C
        the_sim = merge_sim(C, j, max_i, max_m)
        C[max_i][j] = the_sim
        C[j][max_i] = the_sim
        
    I[max_m] = 0 #update I
    
    if np.sum(I) in K:
        temp= sorted([sorted(c) for c in result if c is not None])
        basic_result.append(temp)

basic_result

# basic method of HAC
K_ = [20, 13, 8]

for k in range(len(K_)):
    with open('C:/Users/user/R08725008/'+str(K_[k])+'.txt', 'w') as f:
        for i in range(len(basic_result[k])):
            for j in range(len(basic_result[k][i])):
                f.write(str(basic_result[k][i][j]+1)+'\n')
            f.write('\n')
        f.close()
    
