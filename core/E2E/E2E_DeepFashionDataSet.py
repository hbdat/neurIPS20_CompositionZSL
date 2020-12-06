# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 12:13:14 2020

@author: badat
"""

import os,sys
#import scipy.io as sio
import torch
import numpy as np
import h5py
import time
import pickle
from sklearn import preprocessing
from global_setting_Pegasus import NFS_path_AoA
from torchvision import transforms
from PIL import Image
from threading import Thread
from torch.utils.data import Dataset, DataLoader
from core.ImageTransformation import data_transforms
import torch.nn.functional as F 
#%%
import scipy.io as sio
import pandas as pd
#%%
import pdb
#%%
dataset = "DeepFashion"
img_dir = os.path.join(NFS_path_AoA,'data/DeepFashion/')
anno_path = os.path.join(NFS_path_AoA,'data/DeepFashion/annotation.pkl')

input_size = 224

# cannot load anything into GPU since this class will be used in multi-thread setting which is not compatible with GPU

class E2E_DeepFashionDataSet(Dataset):
    def __init__(self, data_path, is_scale = False, is_unsupervised_attr = False,is_balance =True, split = "train", is_augment = True):
        assert split in ["train","test_seen","test_unseen"]
        
        print(data_path)
        sys.path.append(data_path)
        
        
        self.transform = data_transforms
        self.data_path = data_path
        self.dataset = dataset
        print('$'*30)
        print(self.dataset)
        print('$'*30)
        self.datadir = self.data_path + 'data/{}/'.format(self.dataset)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.is_scale = is_scale
        self.is_balance = is_balance
        self.split = split
        if is_augment == True:
            self.preprocess = "augment"
        else:
            self.preprocess = "deterministic"
        
        if self.is_balance:
            print('Balance dataloader')
        self.is_unsupervised_attr = is_unsupervised_attr
        
        ### load hdf5 ####
        path= self.datadir + 'feature_map_ResNet_101_{}_sep_seen_samples.hdf5'.format(self.dataset)
        print('_____')
        print(path)
        tic = time.clock()
        
        
        hf = h5py.File(path, 'r')
        
        
        att = np.array(hf.get('att'))
        
        ## remap classes this is because there is some classes that does not have training sample
        self.available_classes = np.where(np.sum(att,axis = 1)!=0)[0]
        self.map_old2new_classes = np.ones(att.shape[0])*-1
        self.map_old2new_classes[self.available_classes] = np.arange(self.available_classes.shape[0])
        self.map_old2new_classes = torch.from_numpy(self.map_old2new_classes).long()
        ##
        
        self.att = torch.from_numpy(att).float()
        
        self.normalize_att = torch.tensor([-1])
        
        w2v_att = np.array(hf.get('w2v_att'))
        self.w2v_att = torch.from_numpy(w2v_att).float()
        
        labels = hf['label_train'] #this is a dictionary structure
        seenclasses = [int(l) for l in labels]
        n_sample_classes = [len(labels[str(l)]) for l in seenclasses]
        
        
        test_unseen_label = torch.from_numpy(np.array(hf.get('label_test_unseen'),dtype=np.int32)).long()
        test_seen_label = torch.from_numpy(np.array(hf.get('label_test_seen'),dtype=np.int32)).long()
        
        self.seenclasses = torch.tensor(seenclasses)
        self.unseenclasses = torch.unique(test_unseen_label)
        self.ntrain = sum(n_sample_classes)
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        ## containing missing classes therefore cannot determine the set of all available label
    
        print('Finish loading data in ',time.clock()-tic)
        
        
        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['labels']= labels

#        input('Debug version b')
        self.data['train_unseen'] = {}
        self.data['train_unseen']['labels'] = None
        
        self.data['test_seen'] = {}
        self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['labels'] = test_unseen_label
        ### load hdf5 ####
        
        ### loading image path ###
        self.package = pickle.load(open(anno_path,'rb'))
        self.attr_name = self.package['att_names']
        self.image_files = self.package['image_names']
        self.cat_names = self.package['cat_names']
        
        def convert_path(image_files,img_dir):
            new_image_files = []
            for idx in range(len(image_files)):
                image_file = image_files[idx]
                image_file = os.path.join(img_dir,image_file)
                new_image_files.append(image_file)
            return np.array(new_image_files)
        self.image_files = convert_path(self.image_files,img_dir)
        
        test_seen_loc = np.array(hf.get('test_seen_loc'))
        test_unseen_loc = np.array(hf.get('test_unseen_loc'))
    
        self.data['train_seen']['img_path'] = hf['img_train']#self.image_files[trainval_loc]
        self.data['test_seen']['img_path'] = self.image_files[test_seen_loc]
        self.data['test_unseen']['img_path'] = self.image_files[test_unseen_loc]
    
        ### loading image path ###
        
        self.convert_new_classes()
        
        hf.close()
    
    
    def convert_new_classes(self):
        
        self.dict_train = {}
        for l in self.seenclasses.numpy().tolist():
            assert np.unique(self.data['train_seen']['labels'][str(l)])[0] == l
            self.dict_train[self.map_old2new_classes[l].item()] = self.data['train_seen']['img_path'][str(l)].value.tolist()
        
        self.att = self.att[self.available_classes]
        self.att = F.normalize((self.att+1)/2)
        
        self.data['test_seen']['labels'] = self.map_old2new_classes[self.data['test_seen']['labels']]
        self.data['test_unseen']['labels'] = self.map_old2new_classes[self.data['test_unseen']['labels']]
        
        self.seenclasses = self.map_old2new_classes[self.seenclasses]
        self.unseenclasses = torch.unique(self.data['test_unseen']['labels'])
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        
        
    
    def __len__(self):
        if self.split == "train":
            return self.ntrain
        elif self.split == "test_seen":
            return self.ntest_seen
        elif self.split == "test_unseen":
            return self.ntest_unseen
        else:
            raise Exception("Unknown split")
    
    def __getitem__(self, idx):
        if self.split == "train":
            l = self.seenclasses[idx%self.ntrain_class].item()
            idx_select = np.random.choice(len(self.dict_train[l]), 1).squeeze()
            
            img_file = self.dict_train[l][idx_select]
            img_file = os.path.join(img_dir,img_file)
            label = l
            
        elif self.split == "test_seen":
            img_file = self.data['test_seen']['img_path'][idx]
            label =  self.data['test_seen']['labels'][idx]
            
        elif self.split == "test_unseen":
            img_file = self.data['test_unseen']['img_path'][idx]
            label =  self.data['test_unseen']['labels'][idx]
            
        else:
            raise Exception("Unknown split")
        
        image = Image.open(img_file)
        if image.mode != 'RGB':
            image=image.convert('RGB')
        
        if self.split == "train":
            image = self.transform[self.preprocess](image)
        else:
            image = self.transform["deterministic"](image)
        att = self.att[label]
        
        return label, image, att