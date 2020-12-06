#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 17:21:18 2019

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
import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models.resnet as models
from PIL import Image
import h5py
import numpy as np
import scipy.io as sio
import pickle
from global_setting_Pegasus import NFS_path
#%%
import pdb
#%%
idx_GPU = 6
is_save = True
dataset = 'FLO'
#%%
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
#%%
img_dir = os.path.join(NFS_path,'data/{}/'.format(dataset))
file_paths = os.path.join(NFS_path,'data/xlsa17/data/{}/res101.mat'.format(dataset))
save_path = os.path.join(NFS_path,'data/{}/feature_map_ResNet_101_{}.hdf5'.format(dataset,dataset))
attribute_path = './w2v/{}_attribute.pkl'.format(dataset)
#pdb.set_trace()
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Batch size for training (change depending on how much memory you have)
batch_size = 32

device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")
#%%

model_ref = models.resnet101(pretrained=True)
model_ref.eval()

model_f = nn.Sequential(*list(model_ref.children())[:-2])
model_f.to(device)
model_f.eval()

for param in model_f.parameters():
    param.requires_grad = False
#%%
class CustomedDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_dir , file_paths, transform=None):
        self.matcontent = sio.loadmat(file_paths)
        self.image_files = np.squeeze(self.matcontent['image_files'])
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx][0]
        image_file = os.path.join(self.img_dir,
                                  '/'.join(image_file.split('/')[7:]))
        image = Image.open(image_file)
        if image.mode != 'RGB':
            image=image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

#%%
input_size = 224
data_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
AWA2Dataset = CustomedDataset(img_dir , file_paths, data_transforms)
dataset_loader = torch.utils.data.DataLoader(AWA2Dataset,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=4)
#%%
#with torch.no_grad():
all_features = []
for i_batch, imgs in enumerate(dataset_loader):
    print(i_batch)
    imgs=imgs.to(device)
    features = model_f(imgs)
    all_features.append(features.cpu().numpy())
all_features = np.concatenate(all_features,axis=0)
#%% get remaining metadata
matcontent = AWA2Dataset.matcontent
labels = matcontent['labels'].astype(int).squeeze() - 1

split_path = os.path.join(NFS_path,'data/xlsa17/data/{}/att_splits.mat'.format(dataset))
matcontent = sio.loadmat(split_path)
trainval_loc = matcontent['trainval_loc'].squeeze() - 1
train_loc = matcontent['train_loc'].squeeze() - 1 #--> train_feature = TRAIN SEEN
val_unseen_loc = matcontent['val_loc'].squeeze() - 1 #--> test_unseen_feature = TEST UNSEEN
test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
original_clss_semantics = matcontent['att'].T

#%% class semantic decomposition
original_clss_semantics = torch.tensor(original_clss_semantics).float()
            
U, s, V = torch.svd(original_clss_semantics)
reconstruct = torch.mm(torch.mm(U,torch.diag(s)),torch.transpose(V,1,0))
print('sanity check: {}'.format(torch.norm(reconstruct-original_clss_semantics).item()))

print('shape U:{} V:{}'.format(U.size(),V.size()))
print('s: {}'.format(s))

w2v_att = torch.transpose(V,1,0).cpu().numpy()
att = torch.mm(U,torch.diag(s)).cpu().numpy()
#%%
pdb.set_trace()
if is_save:
    f = h5py.File(save_path, "w")
    f.create_dataset('feature_map', data=all_features,compression="gzip")
    f.create_dataset('labels', data=labels,compression="gzip")
    f.create_dataset('trainval_loc', data=trainval_loc,compression="gzip")
    f.create_dataset('train_loc', data=train_loc,compression="gzip")
    f.create_dataset('val_unseen_loc', data=val_unseen_loc,compression="gzip")
    f.create_dataset('test_seen_loc', data=test_seen_loc,compression="gzip")
    f.create_dataset('test_unseen_loc', data=test_unseen_loc,compression="gzip")
    f.create_dataset('original_clss_semantics', data=original_clss_semantics,compression="gzip")
    f.create_dataset('att', data=att,compression="gzip")
    f.create_dataset('w2v_att', data=w2v_att,compression="gzip")
    f.close()