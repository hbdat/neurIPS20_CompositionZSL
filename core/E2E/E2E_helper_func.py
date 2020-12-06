# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 18:52:17 2020

@author: Warmachine
"""

import torch
import numpy as np
#%% visualization package
from scipy import ndimage
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import skimage.transform
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#%%
import pandas as pd
#%%
import pdb

def E2E_val_gzsl(dataset_loader, test_label, target_classes,in_package,bias = 0):
    device = in_package['device']
    model = in_package['model']
    with torch.no_grad():
        start = 0
        predicted_label = torch.LongTensor(test_label.size())
        for i_batch, imgs in enumerate(dataset_loader):
            
            end = start+imgs.size(0)
            imgs = imgs.to(device)
            out_package = model(imgs)
            
#            if type(output) == tuple:        # if model return multiple output, take the first one
#                output = output[0]
            if isinstance(out_package,dict):
                output = out_package['S_pp']
            else:
                output = out_package
            output[:,target_classes] = output[:,target_classes]+bias
            predicted_label[start:end] = torch.argmax(output.data, 1)

            start = end

        acc = compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, in_package)
        return acc

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size()).fill_(-1)
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label

def E2E_val_zs_gzsl(dataset_loader, test_label, unseen_classes,in_package,bias = 0):
    device = in_package['device']
    model = in_package['model']
    with torch.no_grad():
        start = 0
        predicted_label_gzsl = torch.LongTensor(test_label.size())
        predicted_label_zsl = torch.LongTensor(test_label.size())
        predicted_label_zsl_t = torch.LongTensor(test_label.size())
        for i_batch, imgs in enumerate(dataset_loader):

            end = start+imgs.size(0)
            imgs = imgs.to(device)
            out_package = model(imgs)
            
#            if type(output) == tuple:        # if model return multiple output, take the first one
#                output = output[0]
#           
            if isinstance(out_package,dict):
                output = out_package['S_pp']
            else:
                output = out_package
            
            output_t = output.clone()
            output_t[:,unseen_classes] = output_t[:,unseen_classes]+torch.max(output)+1
            predicted_label_zsl[start:end] = torch.argmax(output_t.data, 1)
            predicted_label_zsl_t[start:end] = torch.argmax(output.data[:,unseen_classes], 1) 
            
            output[:,unseen_classes] = output[:,unseen_classes]+bias
            predicted_label_gzsl[start:end] = torch.argmax(output.data, 1)
            
            start = end
            
        acc_gzsl = compute_per_class_acc_gzsl(test_label, predicted_label_gzsl, unseen_classes, in_package)
        acc_zs = compute_per_class_acc_gzsl(test_label, predicted_label_zsl, unseen_classes, in_package)
        acc_zs_t = compute_per_class_acc(map_label(test_label, unseen_classes), predicted_label_zsl_t, unseen_classes.size(0))
        
        if not np.abs(acc_zs - acc_zs_t) < 0.001:
            print("!!!!!!! MISMATCH IN ZSL score !!!!!")
        #print('acc_zs: {} acc_zs_t: {}'.format(acc_zs,acc_zs_t))
        return acc_gzsl,acc_zs_t

def compute_per_class_acc(test_label, predicted_label, nclass):
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    for i in range(nclass):
        idx = (test_label == i)
        acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx).float()
    return acc_per_class.mean().item()
    
def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, in_package):

    device = in_package['device']
    per_class_accuracies = torch.zeros(target_classes.size()[0]).float().to(device).detach()

    predicted_label = predicted_label.to(device)

    for i in range(target_classes.size()[0]):

        is_class = test_label == target_classes[i]

        per_class_accuracies[i] = torch.div((predicted_label[is_class]==test_label[is_class]).sum().float(),is_class.sum().float())
#        pdb.set_trace()
    return per_class_accuracies.mean().item()

class CustomedDataset(Dataset):
    def __init__(self, image_files,device, transform):
        self.image_files = image_files
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = Image.open(image_file)
        if image.mode != 'RGB':
            image=image.convert('RGB')
        image = self.transform(image)
        #image = image.to(self.device) #error cannot move to device in dataloader
        return image


def E2E_eval_zs_gzsl(dataset,model,device,bias_seen,bias_unseen):
    model.eval()
    batch_size = 200
    num_worker = 10
    print('bias_seen {} bias_unseen {}'.format(bias_seen,bias_unseen))
    
    seen_dataset = CustomedDataset(dataset.data['test_seen']['img_path'],device, dataset.transform["deterministic"])
    seen_dataset_loader = torch.utils.data.DataLoader(seen_dataset,
                                                 batch_size=batch_size, shuffle=False,
                                                 num_workers=num_worker)
    test_seen_label = dataset.data['test_seen']['labels'].to(device)
    
    unseen_dataset = CustomedDataset(dataset.data['test_unseen']['img_path'],device, dataset.transform["deterministic"])
    unseen_dataset_loader = torch.utils.data.DataLoader(unseen_dataset,
                                                 batch_size=batch_size, shuffle=False,
                                                 num_workers=num_worker)
    test_unseen_label = dataset.data['test_unseen']['labels'].to(device)
    
    seenclasses = dataset.seenclasses.to(device)
    unseenclasses = dataset.unseenclasses.to(device)
    
    batch_size = 100
    
    in_package = {'model':model,'device':device, 'batch_size':batch_size}
    
    with torch.no_grad():
        acc_seen = E2E_val_gzsl(seen_dataset_loader, test_seen_label, seenclasses, in_package,bias=bias_seen)
        acc_novel,acc_zs = E2E_val_zs_gzsl(unseen_dataset_loader, test_unseen_label, unseenclasses, in_package,bias = bias_unseen)

    if (acc_seen+acc_novel)>0:
        H = (2*acc_seen*acc_novel) / (acc_seen+acc_novel)
    else:
        H = 0
        
    return acc_seen, acc_novel, H, acc_zs