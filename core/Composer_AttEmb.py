# -*- coding: utf-8 -*-  
"""  
Created on Mon Apr  6 17:52:51 2020  
  
@author: badat  
"""  
  
import torch     
import torch.nn as nn     
import torch.nn.functional as F     
import numpy as np     
from omp.omp import omp  
from core.AttributeEmbed import AttributeEmbed  
import random
import pdb   
import matplotlib.pyplot as plt

#####     
# einstein sum notation     
# b: Batch size \ f: dim feature \ v: dim w2v \ r: number of region \ k: number of classes     
# i: number of attribute \ h : hidden attention dim     
#####     
class Composer_AttEmb(nn.Module):     
    def __init__(self, device,base_model, dim_f=2048,k=3,is_optimize=False,mode_percent = 0.7,is_correct = True,n_comp=10,T=5):   
        super(Composer_AttEmb, self).__init__()   
        self.n_att = base_model.att.size(1)   
        self.dim_f = dim_f   
        self.max_a = None   
        self.a_supp = None   
        self.k=k  
        self.is_optimize = is_optimize  
        self.mode_percent = mode_percent
        self.is_correct = is_correct
        self.n_comp=n_comp
        self.T = T
        
        
        self.att = base_model.att
        print("Attribute Embed")  
        self.attrEmb =  AttributeEmbed(device,dim_f,base_model)  
          
        if self.k > 0:  
            print("Using OMP")  
        else:  
            print("No OMP")  
          
        print("nearest neighbour k {}".format(k))  
        print("T {}".format(self.T))
            
        print("n_comp {}".format(self.n_comp))
#        print("Additional random combination search space {}".format(self.n_aug))   
#        print("WARINING TESTING DUMMY COMBINATION")   
      
    def get_candidate(self,att_s,att_t):  
        ## get composition candidate ##  
        target_att = att_t.cpu().numpy()[np.newaxis].T  
        supp_atts = att_s.cpu().numpy().T  
        res = omp(supp_atts, target_att, nonneg=True, ncoef=self.k, maxit=200, tol=1e-3, ztol=1e-12, verbose=False)  
        active_set = res.active  
        ## get composition candidate ##  
          
        return active_set
       
    def sanity_check(self,P,edit_candidate,log_prob):   
        one_hot = torch.eye(P.size(1)).to(P.device)   
        log_softmax = F.log_softmax(P,dim=1)   
        return torch.abs(log_prob-torch.sum(torch.einsum("iz,iz->i",one_hot[edit_candidate],log_softmax))).item() <= 1e-3   
           
    
    def compose(self,Hs,labels_s,labels_t,Alphas_s=None):           #Hs [zif], labels_s [z], labels_t[b] 
        with torch.no_grad():    
            self.sparse_structure = torch.zeros(1)    
            self.atts_comp = []  
            
            atts_s = self.att[labels_s]  
            atts_t = self.att[labels_t]
            
            Hs_p = []   
            self.max_a = []   
            self.a_supp = []   
            self.hist = []   
            self.Alphas_t = []
            self.M = []
            self.S = []
            
            for b in range(atts_t.size(0)):   
                target_att = atts_t[b]  #[i]
                target = labels_t[b]
                if self.k > 0:  
                    active_set = self.get_candidate(atts_s,target_att)        #[a]  
                    Hs_supp = Hs[active_set]
                    atts_supp = atts_s[active_set]
                else:  
                    active_set = np.arange(Hs.size(0)) 
                    Hs_supp = Hs
                    atts_supp = atts_s
                
                ### Need to sample a multinomial distribution ###
                multinomial_prob = torch.einsum("i,ai->a",target_att,atts_supp)
                multinomial_prob = F.softmax(multinomial_prob*self.T,dim=0)#multinomial_prob = F.softmax(multinomial_prob*self.k,dim=0)
                #multinomial_prob/=torch.sum(multinomial_prob);
                
                multinomial_count = (multinomial_prob*self.n_att).long()
                
                multinomial_count[0] = self.n_att - torch.sum(multinomial_count[1:])
                
                
                combinations = [] 

                categorical_dis = torch.distributions.categorical.Categorical(probs= torch.tensor(multinomial_prob))
                combinations = categorical_dis.sample(sample_shape=torch.Size([self.n_att,self.n_comp]))         #[u] with u is the number of samples
                
                one_hot = torch.eye(len(active_set)).to(Hs_supp.device)      #[aa]  
                
                
                Hs_comp = torch.einsum("iua,aif->uif",one_hot[combinations],Hs_supp)   
                
                prob = self.attrEmb(Hs_comp)    #[uk]
                
                log_prob_dis = torch.log(prob[:,target]) #[u]
                
                log_prob_prior = torch.sum(categorical_dis.log_prob(combinations),dim=0) #[iu]
                
                log_prob = log_prob_dis+log_prob_prior
                
                idx_best_comp = torch.argmax(log_prob)#torch.argmax(prob[:,target])
                
                
                H_p = Hs_comp[idx_best_comp]
                    
                Hs_p.append(H_p[None,:,:])   
                   
                   
            Hs_p = torch.cat(Hs_p,dim=0)    
            
            return Hs_p