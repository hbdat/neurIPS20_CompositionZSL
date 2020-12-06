# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:24:56 2019

@author: Warmachine
"""
  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import numpy as np
import torch.autograd as autograd
#%%  
import pdb  

#####  
# einstein sum notation  
# b: Batch size \ f: dim feature \ v: dim w2v \ r: number of region \ k: number of classes  
# i: number of attribute \ h : hidden attention dim  
#####  
class Margin(nn.Module):
    def __init__(self,base_model, bias = True,no_grad_base = True,is_conservative = True,second_order = False,activate = True,margin_only = False):
        super(Margin, self).__init__()
        self.base_model = [base_model]
        self.nclass = base_model.nclass
        self.device = base_model.device
        
        self.seenclass = base_model.seenclass  
        self.unseenclass = base_model.unseenclass 
        self.is_conservative = is_conservative
        self.second_order = second_order
        self.no_grad_base = no_grad_base
        
        self.activate = activate
        
        self.margin_only = margin_only
        
        
        if self.margin_only:
            print("optimize margin constant")
            self.margin = nn.Parameter(nn.init.zeros_(torch.empty(1)).to(self.device))
        else:
            if self.second_order:
                self.linear1 = nn.Linear(in_features = self.nclass**2,
                                         out_features = 1,
                                         bias=bias).to(self.device)
            else:
                self.linear1 = nn.Linear(in_features = self.nclass,
                                         out_features = 1,
                                         bias=bias).to(self.device)
                
#        if not self.no_grad_base:
#            self.W_1 = base_model.W_1
#            self.V = base_model.V
        
        self.weight_ce = torch.eye(self.nclass).float().to(self.device)
        self.log_softmax_func = nn.LogSoftmax(dim=1)  
        
        self.distributed_bias = torch.ones(1,self.nclass).to(self.device)
        self.distributed_bias[:,self.seenclass] = -1
            
        if self.no_grad_base:
            print("No grad computation for base model")
        else:
            print("Compute gradient of base model")
            
        if self.second_order:
            print("Use second order statistic for progressive model")
        else:
            print("Use class probability only")
        
        print("Calibrate base model prediction")
        
    def compute_attribute_embed(self,Hs,activate=None):
        
        S_pp_base = self.base_model[0].compute_attribute_embed(Hs)['S_pp']
        
        if activate == None:
            activate = self.activate
        
        if activate:
            if self.margin_only:
                S_pp_progress = self.distributed_bias*self.margin
            else:
                prob = F.softmax(S_pp_base,dim=1)
                
                if self.second_order:
                    out_prod = torch.einsum('bk,bj->bkj',prob,prob)
                    inp_calibrate = out_prod.view(out_prod.size(0),-1)
                else:
                    inp_calibrate = prob
                    
                '''
                define a neural network in here
                '''
                
                calibrate = self.linear1(inp_calibrate)
        #        hidden = F.relu(hidden)
        #        calibrate = self.linear2(hidden)
                
                S_pp_progress = torch.einsum('bj,jk->bk',calibrate,self.distributed_bias)
        else:
            S_pp_progress = 0
        
        
        if self.base_model is not None:
            S_pp = S_pp_progress + S_pp_base
        
        package = {'S_pp':S_pp,'S_pp_progress':S_pp_progress}  
        
        return package  
        
    def compute_loss_rank(self,in_package,is_conservative=None):  
        # this is pairwise ranking loss  
        batch_label = in_package['batch_label']  
        S_pp = in_package['S_pp']  
        
        if len(in_package['batch_label'].size()) == 1:
            batch_label = self.weight_ce[batch_label] 
        
        batch_label_idx = torch.argmax(batch_label,dim = 1)
        
        if is_conservative is None:
            is_conservative = self.is_conservative
        
        s_c = torch.gather(S_pp,1,batch_label_idx.view(-1,1))  
        if is_conservative:  
            S_seen = S_pp  
        else:  
            S_seen = S_pp[:,self.seenclass]  
            assert S_seen.size(1) == len(self.seenclass)  
        
        s_max,_ = torch.max(S_seen,dim=1)
        margin = 1-(s_c-s_max)
#        margin = 1-(s_c-S_seen)  
        loss_rank = torch.max(margin,torch.zeros_like(margin))  
        loss_rank = torch.mean(loss_rank)  
        return loss_rank  
    
    def compute_aug_cross_entropy(self,in_package,is_conservative = None):  
        batch_label = in_package['batch_label'] 
        
        if is_conservative is None:
            is_conservative = self.is_conservative
        
        if len(in_package['batch_label'].size()) == 1:
            batch_label = self.weight_ce[batch_label]  
        
        S_pp = in_package['S_pp']  
        
        Labels = batch_label
        
        
        if not is_conservative:  
            S_pp = S_pp[:,self.seenclass]  
            Labels = Labels[:,self.seenclass]  
            assert S_pp.size(1) == len(self.seenclass)  
        
        Prob = self.log_softmax_func(S_pp)  
          
        loss = -torch.einsum('bk,bk->b',Prob,Labels)  
        loss = torch.mean(loss)  
        return loss  
    
    def compute_V(self):
        if self.normalize_V:  
            V_n = F.normalize(self.V)
#            V_n = torch.einsum('iv,i->iv',V_n,self.att_strength)
        else:  
            V_n = self.V  
        return V_n
    
    def extract_attention(self,Fs):
        ret = self.base_model[0].extract_attention(Fs)
            
        return ret
    
    def forward(self,Fs,activate=None):
        package_1 = self.extract_attention(Fs)
        Hs = package_1['Hs']
        package_2 = self.compute_attribute_embed(Hs)
        
        package_out = {'S_pp':package_2['S_pp']} 
        
        return package_out
    