# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 21:32:20 2020

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


#read tokens in all documents
tokens_all = ""
for file in next(os.walk('C:/Users/user/R08725008/IRTM/'))[2]:
    f = open('C:/Users/user/R08725008/IRTM/'+file)
    texts = f.read()
    f.close()
    tokens_all += preprocessing(texts)

#print(tokens_all.split(' '))
token_set = set(tokens_all.split(' ')) #extract distinct words
token_list = list(token_set) #order word list
#print(token_list)
for token in token_list: #obervation: the term that less than three words is uaually meaningless e.g. and, or, km, kg...
    if len(token)<3:
        token_list.remove(token)
token_list = sorted(token_list)
print(len(token_list))
token_list

# use term list and  index to a DataFrame
df = pd.DataFrame(pd.Series(token_list),columns=['term'])
#index starts from one
df['t_index'] = df.index+1
#initialize the document frequency of each term
df['df'] = 0
#adjust the order of columns
df = df[['t_index','term','df']]
df

for file in next(os.walk('C:/Users/user/R08725008/IRTM/'))[2]:
    f = open('C:/Users/user/R08725008/IRTM/'+file)
    texts = f.read()
    f.close()
    tokens_doc = preprocessing(texts)
    # Record the document frequency of each term
    for term in token_list:
        if term in tokens_doc:
            df.loc[df['term']==term,'df'] += 1 
df

#optimization: refined the dictionary by filtering out unimportant words
#remove the low frequency words
df_ = df.drop(df[df.df<3].index)
# remove common words which occurs in 90% of  documents
df_ = df_.drop(df_[df_.df>985].index)
df_.reset_index(drop=True,inplace=True)
df_['t_index'] = df_.index + 1
#Construct the final dictionary
df_

df_.to_csv('C:/Users/user/R08725008/dictionary.txt',index=False,header=False,sep=' ')

tf_list = list(df_.term)
for file in next(os.walk('C:/Users/user/R08725008/IRTM/'))[2]:
    tf_idf = df_[['t_index','term']]
    #Initialize term frequency in tf column
    tf_idf['tf'] = 0
    f = open('C:/Users/user/R08725008/IRTM/'+file)
    texts = f.read()
    f.close()
    #Extract tokens of each document
    tokens_all = preprocessing(texts)
    #Calculate tf value:the number of occurrences of the term in the document
    num_of_terms = 0
    for token in tokens_all.split(' '):
        if token in tf_list:
            tf_idf.loc[tf_idf['term']==token,'tf'] += 1
            num_of_terms += 1 
    #Calculate the inverse document frquency 
    idf = np.log10(1095 / df_.df)
    #Calculate the tf-idf unit vector
    tf_idf.tf = (tf_idf.tf / num_of_terms)* idf
    #Remove the row whcih tf_idf.tf is zero(words not in dictionary)
    tf_idf = tf_idf[tf_idf.tf > 0]
    #normalize to unit vector
    tf_idf_sum = np.sum(tf_idf.tf)
    tf_idf.tf = (tf_idf.tf)/tf_idf_sum
    tf_idf = tf_idf.rename(columns={'tf': 'tf-idf'})
    #Remove term column: we only need t_index & tf_idf
    tf_idf.drop('term',axis=1,inplace=True)
    tf_idf.to_csv('C:/Users/user/R08725008/tf_idf/'+file, index=False, header=False, sep=' ')
    with open('C:/Users/user/R08725008/tf_idf/'+file,'r') as fo: unit_vector = fo.read()
    with open('C:/Users/user/R08725008/tf_idf/'+file, 'w') as result: result.write(str(len(tf_idf))+'\n'+unit_vector)
               
tf_idf

# d1 = pd.read_csv('C:/Users/user/R08725008/tf_idf/15.txt',names=['t_index','tf-idf'], sep=' ')
# d1 = d1.drop(0) 
# d1 = tf_idf.rename(columns={'tf-idf':'tf'})
# sum2 = np.sum(d1.tf)
# print(sum2)

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

sim = similarity('C:/Users/user/R08725008/tf_idf/1.txt','C:/Users/user/R08725008/tf_idf/2.txt')
print("The cosine similarity of  [document 1] and [document 2] is "+str(sim)) 
