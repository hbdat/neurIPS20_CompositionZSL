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
'''   
Since the data matrix is too big for GPU memory, the operation would be carried out in numpy (CPU)   
'''   
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
      
    def construct_additional_search_space_optimal(self,a_supp,Hs_supp):     #[i(z-1)],  [(z-1)if]   
        edit_candidate = torch.argmax(a_supp,dim=1)   
        one_hot = torch.eye(a_supp.size(1)).to(Hs_supp.device)   
           
        H_p_a = torch.einsum("iz,zif->if",one_hot[edit_candidate],Hs_supp)[None,:,:]    #[1if]   
        a_supp_a = torch.einsum("iz,iz->i",one_hot[edit_candidate],a_supp)[:,None]      #[i1]         #Matrix inner product   
           
        return H_p_a,a_supp_a   
       
    def construct_additional_search_space_random(self,a_supp,Hs_supp,n_aug=20):     #[i(z-1)],  [(z-1)if]   
        n_att,n_supp = a_supp.size()   
        edit_candidate = np.random.randint(n_supp,size=(n_aug*n_att))  #[(u*i)]   
        one_hot = torch.eye(n_supp).to(Hs_supp.device)                  #[z,z]   
           
        indicator = one_hot[edit_candidate].view((n_aug,n_att,n_supp))     #[uiz]   
           
        H_p_a = torch.einsum("uiz,zif->uif",indicator,Hs_supp)    #[1if]   
        a_supp_a = torch.einsum("uiz,iz->iu",indicator,a_supp)     #[i1]         #Matrix inner product   
           
        return H_p_a,a_supp_a   
       
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
#                multinomial_prob = torch.rand(len(active_set))#(torch.arange(len(active_set))+1).float()
#                multinomial_prob,_= torch.sort(multinomial_prob);
                multinomial_prob = torch.einsum("i,ai->a",target_att,atts_supp)
                multinomial_prob = F.softmax(multinomial_prob*self.T,dim=0)#multinomial_prob = F.softmax(multinomial_prob*self.k,dim=0)
                #multinomial_prob/=torch.sum(multinomial_prob);
                
                multinomial_count = (multinomial_prob*self.n_att).long()
                
                multinomial_count[0] = self.n_att - torch.sum(multinomial_count[1:])
                
                
#                rv_is_0_mode = torch.rand(1).item()<=self.mode_percent
#                if rv_is_0_mode:
#                    multinomial_count_f=torch.zeros(multinomial_count.size());
#                    multinomial_count_f[0]=multinomial_count[-1];
#                    multinomial_count_f[1:]=multinomial_count[torch.randperm(len(active_set)-1)]
#                    multinomial_count = multinomial_count_f.long()
#                else:
#                    multinomial_count = multinomial_count[torch.randperm(len(active_set))]
                
                #multinomial_prob = multinomial_prob/torch.sum(multinomial_prob)#F.softmax(multinomial_prob/3,dim=0) #multinomial_prob/torch.sum(multinomial_prob)#/torch.sum(Semantic_agg)
                
#                multinomial_prob = multinomial_prob[torch.randperm(multinomial_prob.size(0))]
                
#                multinomial_prob = (np.arange(len(active_set))+1)[::-1]
#                multinomial_prob = multinomial_prob/np.sum(multinomial_prob)#[0.38128206, 0.23057693, 0.17384616, 0.13070512, 0.08358974]#        #[a]
#                
#                multinomial_prob = np.array([0.38128206, 0.23057693, 0.17384616, 0.13070512, 0.08358974])
                
#                multinomial_prob = multinomial_prob[:len(active_set)]
#                multinomial_prob = multinomial_prob/np.sum(multinomial_prob)
                
                combinations = [] 
#                for i in range(self.n_comp):
#                    rand_perm = torch.randperm(self.n_att)
#                    start_idx = 0
#                    end_idx = 0
#                    comp = torch.ones(self.n_att)*-1  #[i]
#                    for a in range(len(active_set)):
#                        end_idx += multinomial_count[a]
#                        idx_a = rand_perm[start_idx:end_idx]
#                        comp[idx_a]=a
#                        start_idx+=multinomial_count[a].item()
#                    combinations.append(comp[:,None])
#                combinations = torch.cat(combinations,dim=1).long()      #[iu]
                
#                for i in range(self.n_att):
                categorical_dis = torch.distributions.categorical.Categorical(probs= torch.tensor(multinomial_prob))
                combinations = categorical_dis.sample(sample_shape=torch.Size([self.n_att,self.n_comp]))         #[u] with u is the number of samples
#                    combinations.append(samples_attr[None])
#                combinations = torch.cat(combinations,dim=0).long()      #[iu]               
                
                ### Need to sample a multinomial distribution ###
                
                
                one_hot = torch.eye(len(active_set)).to(Hs_supp.device)      #[aa]  
                
#                pdb.set_trace()
                
                Hs_comp = torch.einsum("iua,aif->uif",one_hot[combinations],Hs_supp)   
                
                prob = self.attrEmb(Hs_comp)    #[uk]
                
                log_prob_dis = torch.log(prob[:,target]) #[u]
                
                log_prob_prior = torch.sum(categorical_dis.log_prob(combinations),dim=0) #[iu]
                
                log_prob = log_prob_dis+log_prob_prior
                
                idx_best_comp = torch.argmax(log_prob)#torch.argmax(prob[:,target])
                
                
                H_p = Hs_comp[idx_best_comp]
                
                edit_candidate = combinations[:,idx_best_comp] #[i]<==[iu]

                att_comp = torch.einsum("ia,ai->i",one_hot[edit_candidate],atts_supp)     
                
                
                self.atts_comp.append(att_comp[None])  
                  
                if Alphas_s is not None:  
                    Alphas_s_supp = Alphas_s              #[zir]<--[zir]  
                    Alphas_s_supp = Alphas_s_supp[active_set]       #[air]<--[zir]  
                      
                    Indicator_basis = torch.eye(Hs.size(0)).to(Hs.device)          #[zz]  
                    Indicator_basis = Indicator_basis[active_set]           #[az]  
                      
                    Indicator_z = torch.einsum("ia,az->iz",one_hot[edit_candidate],Indicator_basis) #active set in z 
                    
                    Alpha_t = torch.einsum("air,ia->ir",Alphas_s_supp,one_hot[edit_candidate])  #attention weight for each attribute feature
                      
                    Alpha_t = torch.einsum("ir,i->ir",Alpha_t,target_att)
                    
                    Alpha_t = torch.einsum("ir,iz->rz",Alpha_t,Indicator_z)  #aggrgate attention weight of each attribute into attention weight of each image  
                      
                    self.Alphas_t.append(Alpha_t[None])     #[1rz]   
                    self.M.append(Indicator_z[None])
                    self.S.append(self.attrEmb.forward_obs(H_p[None],target_att[None])) # [1i1] <== [1if],[1i]
                    
#                if labels_s is not None:
#                    print(classname[labels_t[b]],'<--',classname[labels_s[active_set]])
                
                hist_t = torch.zeros((1,atts_s.size(0))).to(H_p.device)  
                hist_t[:,:len(active_set)] = torch.sum(one_hot[edit_candidate],dim=0,keepdim=True)#hist_t[:,idx_supp[active_set]] = torch.sum(one_hot[edit_candidate],dim=0,keepdim=True)  
                  
                self.sparse_structure += torch.sum(torch.sqrt(torch.sum(one_hot[edit_candidate],dim=0)))   
                self.hist.append(hist_t.cpu())       
                Hs_p.append(H_p[None,:,:])   
                   
                   
            Hs_p = torch.cat(Hs_p,dim=0)   
            self.hist = torch.cat(self.hist,dim=0).numpy()   
            self.atts_comp = torch.cat(self.atts_comp,dim=0)  
              
            if Alphas_s is not None:  
                self.Alphas_t = torch.cat(self.Alphas_t,dim=0)  
                self.M = torch.cat(self.M,dim=0)
                self.S = torch.cat(self.S)
            return Hs_p