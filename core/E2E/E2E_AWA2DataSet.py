# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:00:06 2020

@author: Warmachine
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
#%%
import scipy.io as sio
import pandas as pd
#%%
import pdb
#%%
dataset = 'AWA2'
img_dir = os.path.join(NFS_path_AoA,'data/{}/'.format(dataset))
mat_path = os.path.join(NFS_path_AoA,'data/xlsa17/data/{}/res101.mat'.format(dataset))
attr_path = './attribute/{}/new_des.csv'.format(dataset)

input_size = 224

#data_transforms = {
#    'augment': transforms.Compose([
#        transforms.RandomResizedCrop(input_size),
#        transforms.RandomHorizontalFlip(),
#        transforms.ToTensor(),
#        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#    ]),
#    'deterministic': transforms.Compose([
#        transforms.Resize(256), #transforms.Resize(input_size), #
#        transforms.CenterCrop(input_size),
#        transforms.ToTensor(),
#        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#    ]),
#}


# data_transforms = transforms.Compose([
#     transforms.Resize(input_size),
#     transforms.CenterCrop(input_size),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

class E2E_AWA2DataSet(Dataset):
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
        
        self.attr_name = pd.read_csv(attr_path)['new_des']
        
        if self.is_balance:
            print('Balance dataloader')
        self.is_unsupervised_attr = is_unsupervised_attr
        
        ### load hdf5 ####
        path= self.datadir + 'feature_map_ResNet_101_{}.hdf5'.format(self.dataset)
        print('_____')
        print(path)
        tic = time.clock()
        
        
        hf = h5py.File(path, 'r')
        
        labels = np.array(hf.get('labels'))
        trainval_loc = np.array(hf.get('trainval_loc'))
        test_seen_loc = np.array(hf.get('test_seen_loc'))
        test_unseen_loc = np.array(hf.get('test_unseen_loc'))
        
        print('Expert Attr')
        att = np.array(hf.get('att'))
        
        print("threshold at zero attribute with negative value")
        att[att<0]=0
        
        self.att = torch.from_numpy(att).float()
        
        original_att = np.array(hf.get('original_att'))
        self.original_att = torch.from_numpy(original_att).float()
        
        w2v_att = np.array(hf.get('w2v_att'))
        self.w2v_att = torch.from_numpy(w2v_att).float()
        
        self.normalize_att = self.original_att/100
    
        print('Finish loading data in ',time.clock()-tic)
        
        
        train_label = torch.from_numpy(labels[trainval_loc]).long()
        test_unseen_label = torch.from_numpy(labels[test_unseen_loc])
        test_seen_label = torch.from_numpy(labels[test_seen_loc])

        self.ntrain = train_label.size()[0]
        self.ntest_unseen = test_unseen_label.size()[0]
        self.ntest_seen = test_seen_label.size()[0]
        
        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy()))
        
        
        self.unseenclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy()))
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()

        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['labels']= train_label


        self.data['train_unseen'] = {}
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['labels'] = test_unseen_label
        ### load hdf5 ####
        
        ### partition training data into classes ###
        self.idxs_list = []
        train_label = self.data['train_seen']['labels']
        for c in self.seenclasses:
            idx_c = torch.nonzero(train_label == c).squeeze()
            self.idxs_list.append(idx_c)
        ### partition training data into classes ###
        
        ### loading image path ###
        self.matcontent = sio.loadmat(mat_path)
        self.image_files = np.squeeze(self.matcontent['image_files'])
        
        def convert_path(image_files,img_dir):
            new_image_files = []
            for idx in range(len(image_files)):
                image_file = image_files[idx][0]
                image_file = os.path.join(img_dir,'/'.join(image_file.split('/')[5:]))
                new_image_files.append(image_file)
            return np.array(new_image_files)
        
        self.image_files = convert_path(self.image_files,img_dir)
        
        self.data['train_seen']['img_path'] = self.image_files[trainval_loc]
        self.data['test_seen']['img_path'] = self.image_files[test_seen_loc]
        self.data['test_unseen']['img_path'] = self.image_files[test_unseen_loc]
        ### loading image path ###
        
        hf.close()
    
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
            idx_class = idx%self.ntrain_class
            idxs_samples = self.idxs_list[idx_class]
            idx_select = np.random.choice(len(idxs_samples), 1)
            idx_select_sample = idxs_samples[idx_select].squeeze()
            
            img_file = self.data['train_seen']['img_path'][idx_select_sample]
            label =  self.data['train_seen']['labels'][idx_select_sample]
            
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