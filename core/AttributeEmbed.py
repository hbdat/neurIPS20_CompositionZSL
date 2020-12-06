# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 18:09:08 2020

@author: badat
"""

import torch   
import torch.nn as nn   
import torch.nn.functional as F   
import numpy as np 
import torch.autograd as autograd 
#%%   
import pdb   
 

# einstein sum notation   
# b: Batch size \ f: dim feature \ v: dim w2v \ r: number of region \ k: number of classes   
# i: number of attribute \ h : hidden attention dim   
#####   
class AttributeEmbed(nn.Module): 
    def __init__(self,device,dim_f,base_model,is_optimize = False): 
        super(AttributeEmbed, self).__init__() 
        self.dim_f = dim_f   
        self.init_w2v_att = base_model.init_w2v_att
        self.dim_v = self.init_w2v_att.size(1)   
        self.normalize_V = base_model.normalize_V
        self.nclass = base_model.nclass
        self.seenclass = base_model.seenclass
        ## Inherent from the old model ##
        self.att = base_model.att
        mask_bias = np.ones((1,self.nclass))
        mask_bias[:,self.seenclass.cpu().numpy()] *= -1
        self.mask_bias = torch.tensor(mask_bias).float().to(device)
        self.bias = 1
        ## Inherent from the old model ##
        
        if is_optimize:
            print("Optimize V")
            self.V = nn.Parameter(base_model.V.data.clone().to(device))
        else:
            print("Inherit V")
            self.V = base_model.V#nn.Parameter(base_model.V.data.clone().to(device))#nn.Parameter(self.init_w2v_att.clone().to(device))
        
        print("Repurpose W_1") 
        if is_optimize:
            print("Optimize W")
            self.W_1 = nn.Parameter(base_model.W_1.data.clone().to(device))
        else:
            print("Inherit W_1")
            self.W_1 = base_model.W_1#nn.Parameter(base_model.W_1.data.clone().to(device))#nn.Parameter(nn.init.normal_(torch.empty(self.dim_v,self.dim_f)).to(device)) 
        
    
    def forward(self,Hs):      #[bif,zi]
        B = Hs.size(0)  
        
        V_n = self.compute_V() 
        
        S_p = torch.einsum('iv,vf,bif->bi',V_n,self.W_1,Hs)  
         
        S_pp = torch.einsum('zi,bi->biz',self.att,S_p) 
        
        A_b_p = self.att.new_full(S_p.shape,fill_value = 1)  
        
        S_pp = torch.einsum('ki,bi,bi->bik',self.att,A_b_p,S_p)
        
        S_pp = torch.sum(S_pp,axis=1)        #[bk] <== [bik]
        
        self.vec_bias = self.mask_bias*self.bias
        S_pp = S_pp + self.vec_bias
        
        prob = F.softmax(S_pp,dim=-1)
        
        return prob
    
    def forward_obs(self,Hs,att_t):      #[bif,zi] 
        V_n = self.compute_V()  
         
        S_p = torch.einsum('iv,vf,bif->bi',V_n,self.W_1,Hs)   
          
        S_pp = torch.einsum('zi,bi->biz',att_t,S_p)  
         
        return S_pp 
        
    def compute_V(self): 
        if self.normalize_V:   
            V_n = F.normalize(self.V) 
#            V_n = torch.einsum('iv,i->iv',V_n,self.att_strength) 
        else:   
            V_n = self.V   
        return V_n 