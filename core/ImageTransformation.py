# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:38:59 2020

@author: Warmachine
"""

from torchvision import transforms
#%%

input_size = 224

data_transforms = {
    'augment': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
#        transforms.RandomRotation((-45,45)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'deterministic': transforms.Compose([
        transforms.Resize(256), #transforms.Resize(input_size), #
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
