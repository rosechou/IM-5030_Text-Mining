# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 12:18:44 2020

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

import tqdm
from tqdm import tqdm

import math

import pickle

from collections import Counter
from pandas import DataFrame

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

for file in tqdm(next(os.walk('C:/Users/user/R08725008/IRTM/'))[2]):
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
for file in tqdm(next(os.walk('C:/Users/user/R08725008/IRTM/'))[2]):
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

# dict_all = pd.read_csv('C:/Users/user/R08725008/dictionary.txt', index_col= None, header=None,sep=' ')
# dict_list = list(dict_all[1])
# dict_all.drop(0,axis=1,inplace=True)
# dict_all.columns = ['term','score']
# dict_all.index = dict_all['term']
# dict_all.drop('term',axis=1,inplace=True)
# dict_all

dict_df = pd.read_csv('C:/Users/user/R08725008/dictionary.txt',header=None,index_col=None,sep=' ')
terms = dict_df[1].tolist() #all terms

training_dict = {}
f = open ('C:/Users/user/R08725008/training.txt', "r") 
for line in f:
    items = line.split()
    key, values = int(items[0]), items[1:]
    training_dict.setdefault(key, []).extend(values)
print(training_dict)

with open('C:/Users/user/R08725008/training.txt','r') as f:
    train_id = f.read().splitlines()
train_dict = {}
for trainid in train_id:
    trainid = trainid.split(' ')
    trainid = list(filter(None, trainid))
    train_dict[trainid[0]] = trainid[1:]
print(train_dict) #class:doc_id


in_dir = 'C:/Users/user/R08725008/IRTM/'
# train_dict_ = {}
class_token = []
class_dict = {}
for c,d in train_dict.items():
    for doc in d:
        f = open('C:/Users/user/R08725008/IRTM/'+doc+'.txt')
        texts = f.read()
        f.close()
        tokens_all = preprocessing(texts)
        tokens_all = tokens_all.split(' ')
        tokens_all = list(set(filter(None,tokens_all)))
        class_token.append(tokens_all)
    class_dict[c]=class_token
    class_token=[]
# len(class_dict['1'])
    
    
dict_df.drop(0,axis=1,inplace=True)
dict_df.columns = ['term','score_llr']
dict_df.index = dict_df['term']
dict_df.drop('term',axis=1,inplace=True)
#print(dict_df)

classes = 13
dict_df['score_llr'] = 0
dict_df['score_chi'] = 0

for term in tqdm(terms): 
    scores_llr = []
    scores_chi = []
    c=1
    for _ in range(len(class_dict)): 
        n11=e11=0
        n10=e10=0
        n01=e01=0
        n00=e00=0
        for k,v in class_dict.items():
            if k == str(c): #ontopic
                for r in v:
                    if term in r: #present
                        n11+=1
                    else: #absent
                        n10+=1

            else: #off topic
                for r in v:
                    if term in r:
                        n01+=1
                    else:
                        n00+=1

        c+=1
        n11+=1e-10
        n10+=1e-10
        n01+=1e-10
        n00+=1e-10 
        
        #chi-squre
        N = n11+n10+n01+n00 
        e11 = N * (n11+n01)/N * (n11+n10)/N
        e10 = N * (n11+n10)/N * (n10+n00)/N
        e01 = N * (n11+n01)/N * (n01+n00)/N
        e00 = N * (n01+n00)/N * (n10+n00)/N
        score_chi = ((n11-e11)**2)/e11 + ((n10-e10)**2)/e10 + ((n01-e01)**2)/e01 + ((n00-e00)**2)/e00
        scores_chi.append(score_chi)
        
        #LLR
        N = n11+n10+n01+n00
        score_llr = (((n11+n01)/N) ** n11) * ((1 - ((n11+n01)/N)) ** n10) * (((n11+n01)/N) ** n01) * ((1 - ((n11+n01)/N)) ** n00)
        score_llr /= ((n11/(n11+n10)) ** n11) * ((1 - (n11/(n11+n10))) ** n10) * ((n01/(n01+n00)) ** n01) * ((1 - (n01/(n01+n00))) ** n00)
        score_llr = -2 * math.log(score_llr, 10) 
        scores_llr.append(score_llr)
        

    dict_df.loc[term,'score_llr'] = np.mean(scores_llr)
    dict_df.loc[term,'score_chi'] = np.mean(scores_chi)
print(dict_df)    

df2=dict_df.sort_values(by='score_llr', ascending=False).reset_index().head(100)
df3=dict_df.sort_values(by='score_chi', ascending=False).reset_index().head(140)

feature = list(set(df2.term.tolist()))
feature2= list(set(df3.term.tolist()))
feature.extend(feature2)
feature= list(set(feature))
len(feature)

with open('C:/Users/user/R08725008/training.txt','r') as f:
    train_id = f.read().splitlines()
train_dict = {}
train_id

#id with class docs
for i in train_id:
    i = i.split(' ')
    i = list(filter(None, i))
    train_dict[i[0]] = i[1:] #first digit is the class label

in_dir = 'C:/Users/user/R08725008/IRTM/'
train_dict_ = {}
class_token = []
class_dict = {}
train1 = []
train2= []
train_ids = []
for c,d in tqdm(train_dict.items()):
    for doc in d:
        train_ids.append(doc)
        trainX = np.array([0]*len(feature))
        f = open('C:/Users/user/R08725008/IRTM/'+doc+'.txt')
        texts = f.read()
        f.close()
        tokens_all = preprocessing(texts)
        tokens_all = tokens_all.split(' ')
        tokens_all = Counter(tokens_all)
        for key,value in tokens_all.items():
            if key in feature:
                trainX[feature.index(key)] = int(value)

        train1.append(trainX)
        train2.append(int(c))
        
train1 = np.array(train1)
train2 = np.array(train2)


#term index matrix
tokens_all_class=[]
matrix=[]
for c,d in tqdm(train_dict.items()):
    for doc in d:
        f = open('C:/Users/user/R08725008/IRTM/'+doc+'.txt')
        texts = f.read()
        f.close()
        tokens_all = preprocessing(texts)
        tokens_all = tokens_all.split(' ')
        tokens_all = list(filter(None,tokens_all))
        tokens_all_class.extend(tokens_all)
    tokens_all = Counter(tokens_all_class)
    matrix.append(tokens_all)
    

def trainMultinomialNB(train_set=train_dict,term_list=feature,matrix=matrix):
    prior = np.zeros(len(train_set))
    cond_prob = np.zeros((len(train_set), len(term_list)))
    
    for i,docs in train_set.items(): #13 classes
        prior[int(i)-1] = len(docs)/len(train_ids) 
        token_count=0
        tf = np.zeros(len(term_list))
        for idx,term in enumerate(term_list):
            try:
                tf[idx] = matrix[int(i)-1][term]  
            except:
                token_count+=1

        tf = tf + np.ones(len(term_list)) #add on smothing
        tf = tf/(sum(tf) +token_count) 
        cond_prob[int(i)-1] = tf 
    return prior, cond_prob

prior,cond_prob = trainMultinomialNB()

def ApplyMultinomialNB(test_id,prob=False,prior=prior,cond_prob=cond_prob,term_list=feature):
    f = open('C:/Users/user/R08725008/IRTM/'+str(test_id)+'.txt')
    texts = f.read()
    f.close()
    tokens_all = preprocessing(texts)
    tokens_all = tokens_all.split(' ')
    tokens_all = list(filter(None,tokens_all))
    
    class_matrix = []
    for i in range(13):
        val=0
        val += math.log(prior[i],10)
        for token in tokens_all:
            if token in term_list:
                val += math.log(cond_prob[i][term_list.index(token)])
        class_matrix.append(val)
    if prob:
        return np.array(class_matrix)
    else:
        return(np.argmax(class_matrix)+1)
        
with open('C:/Users/user/R08725008/training.txt','r') as f:
    train_id = f.read().splitlines()
train_dict = {}
test_id = []
train_ids=[]
for trainid in train_id:
    trainid = trainid.split(' ')
    trainid = list(filter(None, trainid))
    train_ids.extend(trainid[1:])
for i in range(1095):
    if str(i+1) not in train_ids:
        test_id.append(i+1)
ans=[]
for doc in tqdm(test_id):
    ans.append(ApplyMultinomialNB(doc))
print(ans)
res = pd.DataFrame(list(zip(test_id,ans)),columns=['id','Value'])
res.to_csv('C:/Users/user/R08725008/res.csv',index=False)
#res

