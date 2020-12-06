#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:01:03 2019

@author: war-machince
"""
import os,sys
pwd = os.getcwd()
sys.path.insert(0,pwd)
#%%
print('-'*30)
print(os.getcwd())
print('-'*30)
#%%
import pdb
import pandas as pd
import numpy as np
import gensim.downloader as api
import scipy.io as sio
import pickle
from global_setting_Pegasus import NFS_path
#%%
print('Loading pretrain w2v model')
model_name = 'word2vec-google-news-300'#best model
model = api.load(model_name)
dim_w2v = 300
print('Done loading model')
#%%
dataset = 'AWA2'
#%% extract w2v class name
file_path = NFS_path+'/data/xlsa17/data/{}/att_splits.mat'.format(dataset)
matcontent = sio.loadmat(file_path)
classnames = matcontent['allclasses_names'].flatten()
classnames = [' '.join(i.item().split('+')) for i in classnames]
#%%
counter_err = 0
all_class_w2v = []
for s in classnames:
    print(s)
    words = s.split(' ')
    if words[-1] == '':     #remove empty element
        words = words[:-1]
    w2v = np.zeros(dim_w2v)
    for w in words:
        try:
            w2v += model[w]
        except Exception as e:
            print(e)
            counter_err += 1
    all_class_w2v.append(w2v[np.newaxis,:])
print('counter_err ',counter_err)

all_class_w2v = np.concatenate(all_class_w2v)
with open('./w2v/{}_class.pkl'.format(dataset),'wb') as f:
    pickle.dump(all_class_w2v,f)    