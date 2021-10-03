#!/usr/bin/env python
# coding: utf-8

# In[16]:


# install NLTK  
import nltk
#import string
# install related NLTK packages 
nltk.download()
# Porter’s algorithm
from nltk.stem.porter import *
#from nltk.corpus import stopwords


# In[2]:


# stop_words = set(stopwords.words('english'))
# print(stop_words)


# In[17]:


# read the data 
file = open('C:/Users/user/R08725008/28.txt','r')
texts = file.read()
texts = texts.translate(str.maketrans('', '', "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"))
#print(string.punctuation)
#texts = file.read().replace(',', '').replace('.', '').replace(';', '').replace('\'s', '')
print(texts)
print("=============================================================================================================")


# In[18]:


#Tokenization
word_tokenize = texts.split()
#print(word_tokenize)


# In[22]:


# Stemming using Porter’s algorithm
ps = PorterStemmer() 
# Stopword lists
stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 
              'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 
              'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 
              'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 
              'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over',
              'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those',
              'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', '.', ',', '"', "'", 
              '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '\'s','\'m','\'re','\'ll','\'d','n\'t','shan\'t','thats']


# In[23]:


# for i in word_tokenize:
#     print(i.lower())


# In[24]:


#Lowercasing everything
tokens = [i.lower() for i in word_tokenize if i.lower() not in stop_words]  #Stopword removal
#print(tokens)
token_result = ''
for i,token in enumerate(tokens):
    if i != len(tokens)-1: # not leave empty in the end of file
        token_result += ps.stem(token) + ' '
    else:
        token_result += ps.stem(token)


# In[26]:


file = open('C:/Users/user/R08725008/result.txt','w')
# Save the result as a txt file
file.write(token_result) 
file.close()
print(token_result)


# In[ ]:




