# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 22:13:45 2019

@author: Warmachine
"""

import os,sys
pwd = os.getcwd()
sys.path.insert(0,pwd) 
print('-'*30)
print(os.getcwd())
print('-'*30)
#%%
import wikipedia
import pandas as pd
import nltk
import numpy as np
from nltk.tokenize import MWETokenizer
import pdb
import gensim.downloader as api
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from global_setting_Pegasus import NFS_path
import pickle
#%%
nltk.download('stopwords')
nltk.download('punkt')
#%%
stop_words = set(stopwords.words('english')) 
print('Loading pretrain model')
model_name = 'glove-wiki-gigaword-300'#'conceptnet-numberbatch-17-06-300' #
model = api.load(model_name)#api.load("fasttext-wiki-news-subwords-300")
dim_w2v = 300
#%%
dataset = 'CUB'
context_size = 10
file_name = './w2v/{}_contexts_{}_{}.pkl'.format(dataset,context_size,model_name)
#%%
replace_word = [('Clark Nutcracker','Clark\'s nutcracker'),
                ('Nelson Sharp tailed Sparrow','Nelson\'s sparrow'),
                ('Bewick Wren','Bewick\'s wren')]
#%%
path = NFS_path+'data/CUB/CUB_200_2011/CUB_200_2011/classes.txt'
with open(path,"r") as file: 
    classes = file.read().splitlines()
classes = [c.split('.')[-1].replace('_',' ') for c in classes]
#%%
for pair in replace_word:
    for idx,s in enumerate(classes):
        classes[idx]=s.replace(pair[0],pair[1])
#%%
training_data = []
print('Tokenize wiki text')
error = 0
for idx,name in enumerate(classes):#num_class
#    if idx%500==0:
#        with open(file_name,'wb') as f:
#            pickle.dump(training_data,f)
    print('-'*50)
    print(idx,name)
    contexts = []
    try:
        try:
            content=wikipedia.page(name).summary#
        except wikipedia.exceptions.DisambiguationError as ambiguity:
            content=wikipedia.page(ambiguity.options[0]).summary
        
        sentences = content.split('.')[:-1]
        print('ETS :',len(content.split(' '))//context_size)
        for sentence in sentences:
            context = np.zeros(dim_w2v)
            words = word_tokenize(sentence) 
            filtered_words = [w.lower() for w in words if not w.lower() in stop_words]
            n_words = 0.0
            counter = 0
            for idx_w,word in enumerate(filtered_words):
                try:
                    context += model[word]
                    n_words += 1.0
                    counter += 1
                
                    if counter >= context_size or idx_w == len(filtered_words)-1:
                        context /= n_words
                        contexts.append(context[np.newaxis,:])
                        ## reset ##
                        context = np.zeros(dim_w2v)
                        counter = 0
                        n_words = 0.0
                        ## reset ##
                except Exception as e:
#                    pdb.set_trace()
                    print(e)
        print(len(contexts))
    except Exception as e:
        print(e)
        error += 1
        pdb.set_trace()
    
    if len(contexts)>0:
        training_data.append(np.concatenate(contexts))
    else:
        training_data.append(None)

print('error {}'.format(error))
pdb.set_trace()

with open(file_name,'wb') as f:
    pickle.dump([training_data,classes],f)