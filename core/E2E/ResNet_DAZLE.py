# -*- coding: utf-8 -*-  
"""  
Created on Thu Jul  4 17:39:45 2019  
  
@author: badat  
"""  
  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import numpy as np
import torch.autograd as autograd
#%%  
import pdb  
#%%  
class ResNet_DAZLE(nn.Module):  
    #####  
    # einstein sum notation  
    # b: Batch size \ f: dim feature \ v: dim w2v \ r: number of region \ k: number of classes  
    # i: number of attribute \ h : hidden attention dim  
    #####  
    def __init__(self,dim_f,dim_v,  
                 init_w2v_att,att,normalize_att,  
                 seenclass,unseenclass,  
                 lambda_1,lambda_2,lambda_3,
                 device,
                 trainable_w2v = False, normalize_V = False, normalize_F = False, is_conservative = False,
                 prob_prune=0,desired_mass = -1,uniform_att_1 = False,uniform_att_2 = False, is_conv = False,
                 is_bias = False,bias = 1,non_linear_act=False,
                 loss_type = 'CE',non_linear_emb = False,
                 is_sigmoid = False,margin = 1,
                 base_ResNet=None):  
        super(ResNet_DAZLE, self).__init__()  
        self.dim_f = dim_f  
        self.dim_v = dim_v  
        self.dim_att = att.shape[1]  
        self.nclass = att.shape[0]  
        self.hidden = self.dim_att//2
        self.init_w2v_att = init_w2v_att
        self.non_linear_act = non_linear_act
        self.loss_type = loss_type
        self.device = device
        
        self.base_ResNet = base_ResNet
        
        if is_conv:
            r_dim = dim_f//2
            self.conv1 = nn.Conv2d(dim_f, r_dim, 2) #[2x2] kernel with same input and output dims
            print('***Reduce dim {} -> {}***'.format(self.dim_f,r_dim))
            self.dim_f = r_dim
            self.conv1_bn = nn.BatchNorm2d(self.dim_f)
            
            
        if init_w2v_att is None:  
            self.V = nn.Parameter(nn.init.normal_(torch.empty(self.dim_att,self.dim_v)).to(device))  
        else:
            self.init_w2v_att = F.normalize(torch.tensor(init_w2v_att))
            if trainable_w2v:
                self.V = nn.Parameter(self.init_w2v_att.clone().to(device))
            else:
                self.V = self.init_w2v_att.clone().to(device)
        
        #self.tensors.append(self.V)
        
        self.att = F.normalize(torch.tensor(att)).to(device)
        
        self.att_entropy = self.get_attr_entropy(self.att)
        
        self.W_1 = nn.Parameter(nn.init.normal_(torch.empty(self.dim_v,self.dim_f)).to(device)) #nn.utils.weight_norm(nn.Linear(self.dim_v,self.dim_f,bias=False))#  
        self.W_2 = nn.Parameter(nn.init.zeros_(torch.empty(self.dim_v,self.dim_f)).to(device)) #nn.utils.weight_norm(nn.Linear(self.dim_v,self.dim_f,bias=False))#  
        
        ## second layer attenion conditioned on image features
        self.W_3 = nn.Parameter(nn.init.zeros_(torch.empty(self.dim_v,self.dim_f)).to(device))
        
        ## Compute the similarity between classes  
        self.P = torch.mm(self.att,torch.transpose(self.att,1,0))  
        assert self.P.size(1)==self.P.size(0) and self.P.size(0)==self.nclass  
        #weight_ce = construct_pseudo_labels(self.P,unseenclass,preserved_mass = 0.2,n=0)
        self.weight_ce = torch.eye(self.nclass).float().to(device)#nn.Parameter(torch.tensor(weight_ce).float(),requires_grad = False)  
        ## attention level 1  

        self.normalize_V = normalize_V  
        self.normalize_F = normalize_F   
        self.is_conservative = is_conservative  
        self.is_conv = is_conv
        self.is_bias = is_bias
        
        self.seenclass = seenclass  
        self.unseenclass = unseenclass  
        self.normalize_att = normalize_att   
        
#        if is_bias:
        self.bias = torch.tensor(bias).to(device)
        mask_bias = np.ones((1,self.nclass))
        mask_bias[:,self.seenclass.cpu().numpy()] *= -1
        self.mask_bias = torch.tensor(mask_bias).float().to(device)
        
        margin_CE = np.ones((1,self.nclass))
        margin_CE[:,self.seenclass.cpu().numpy()] = margin 
        margin_CE[:,self.unseenclass.cpu().numpy()] = - margin 
        self.margin_CE = torch.tensor(margin_CE).float().to(device)
        
        if desired_mass == -1:  
            self.desired_mass = self.unseenclass.size(0)/self.nclass#nn.Parameter(torch.tensor(self.unseenclass.size(0)/self.nclass),requires_grad = False)#nn.Parameter(torch.tensor(0.1),requires_grad = False)#  
        else:  
            self.desired_mass = desired_mass#nn.Parameter(torch.tensor(desired_mass),requires_grad = False)#nn.Parameter(torch.tensor(self.unseenclass.size(0)/self.nclass),requires_grad = False)#  
        self.prob_prune = torch.tensor(prob_prune).to(device)
         
        self.lambda_1 = lambda_1  
        self.lambda_2 = lambda_2  
        self.lambda_3 = lambda_3  
        self.loss_att_func = nn.BCEWithLogitsLoss()
        self.log_softmax_func = nn.LogSoftmax(dim=1)  
        self.uniform_att_1 = uniform_att_1
        self.uniform_att_2 = uniform_att_2
        
        self.non_linear_emb = non_linear_emb
        
        print('-'*30)  
        print('Configuration')  
        
        print('loss_type {}'.format(loss_type))
        
        if self.is_conv:
            print('Learn CONV layer correct')
        
        if self.normalize_V:  
            print('normalize V')  
        else:  
            print('no constraint V')  
              
        if self.normalize_F:  
            print('normalize F')  
        else:  
            print('no constraint F')  
              
        if self.is_conservative:  
            print('training to exclude unseen class [seen upperbound]')  
        if init_w2v_att is None:  
            print('Learning word2vec from scratch with dim {}'.format(self.V.size()))  
        else:  
            print('Init word2vec')  
        
        if self.non_linear_act:
            print('Non-linear relu model')
        else:
            print('Linear model')
        
        print('loss_att {}'.format(self.loss_att_func))  
        print('Bilinear attention module')  
        print('*'*30)  
        print('Measure w2v deviation')
        if self.uniform_att_1:
            print('WARNING: UNIFORM ATTENTION LEVEL 1')
        if self.uniform_att_2:
            print('WARNING: UNIFORM ATTENTION LEVEL 2')
        print('new Laplacian smoothing with desire mass {} 4'.format(self.desired_mass))  
        #print('Negative log likelihood for MAX unseen 5')
        print('Compute Pruning loss {}'.format(self.prob_prune))  
        if self.is_bias:
            print('Add one smoothing')
#        print('Class agnostic attribute attention with hidden att {}'.format(self.hidden))
        print('Second layer attenion conditioned on image features')
        print('-'*30)  
        
        if self.non_linear_emb:
            print('non_linear embedding')
            self.emb_func = torch.nn.Sequential(
                                torch.nn.Linear(self.dim_att, self.dim_att//2),
                                torch.nn.ReLU(),
                                torch.nn.Linear(self.dim_att//2, 1),
                            )
        
        self.is_sigmoid = is_sigmoid
        if self.is_sigmoid:
            print("Sigmoid on attr score!!!")
        else:
            print("No sigmoid on attr score")                  
    
    def get_attr_entropy(self,att):
        eps = 1e-8
        mass=torch.sum(att,dim = 0,keepdim=True)
        att_n = torch.div(att,mass+eps)
        entropy = torch.sum(-att_n*torch.log(att_n+eps),dim=0)
        assert len(entropy.size())==1
        return entropy
    
    def compute_loss_rank(self,in_package):  
        # this is pairwise ranking loss  
        batch_label = in_package['batch_label']  
        S_pp = in_package['S_pp']  
        
        batch_label_idx = torch.argmax(batch_label,dim = 1)
        
        s_c = torch.gather(S_pp,1,batch_label_idx.view(-1,1))  
        if self.is_conservative:  
            S_seen = S_pp  
        else:  
            S_seen = S_pp[:,self.seenclass]  
            assert S_seen.size(1) == len(self.seenclass)  
          
        margin = 1-(s_c-S_seen)  
        loss_rank = torch.max(margin,torch.zeros_like(margin))  
        loss_rank = torch.mean(loss_rank)  
        return loss_rank  
     
    def compute_loss_pruning(self,in_package):  
        A_p = in_package['A_p']  
        loss = torch.sum(A_p,dim=1)/self.dim_att  
#        loss = torch.abs(loss)  
        loss = torch.abs(loss-self.prob_prune)  
        loss = torch.mean(loss)  
        return loss  
     
    def compute_loss_baseline(self,in_package):  
        batch_label = in_package['batch_label']  
        S_pp = in_package['S_pp']  
        S_b_pp = in_package['S_b_pp']  
        # better than baseline loss  
        batch_label_idx = torch.argmax(batch_label,dim = 1)
        s_c = torch.gather(S_pp,1,batch_label_idx.view(-1,1))  
        s_b_c = torch.gather(S_b_pp,1,batch_label_idx.view(-1,1))  
        margin_b = torch.max(s_b_c-s_c,torch.zeros_like(s_c))  
        loss_baseline = torch.mean(margin_b)  
        return loss_baseline  
      
    def compute_loss_Laplace(self,in_package):  
        S_pp = in_package['S_pp']  
        Prob_all = F.softmax(S_pp,dim=-1)  
        Prob_unseen = Prob_all[:,self.unseenclass]  
        assert Prob_unseen.size(1) == len(self.unseenclass)  
        mass_unseen = torch.sum(Prob_unseen,dim=1)   
        loss_pmp = -torch.mean(torch.log(mass_unseen))
        return loss_pmp  
           
          
    def compute_entropy_with_logits(self,V):  
        e = F.softmax(V, dim=1) * F.log_softmax(V, dim=1)  
        e = -1.0 * torch.sum(e,dim=1)  
        e = torch.mean(e)  
        return e  
     
    def compute_entropy(self,V): 
        mass = torch.sum(V,dim = 1, keepdim = True) 
        V_n = torch.div(V,mass) 
        e = V_n * torch.log(V_n)  
        e = -1.0 * torch.sum(e,dim=1)  
        e = torch.mean(e)  
        return e  
    
    def compute_V(self):
        if self.normalize_V:  
            V_n = F.normalize(self.V)
#            V_n = torch.einsum('iv,i->iv',V_n,self.att_strength)
        else:  
            V_n = self.V  
        return V_n
    
    def compute_aug_cross_entropy(self,in_package,is_conservative = None, override_bias = False):  
        batch_label = in_package['batch_label'] 
        
        if override_bias:
            is_bias = False
        else:
            is_bias = self.is_bias
        
        if is_conservative is None:
            is_conservative = self.is_conservative
        
        if len(in_package['batch_label'].size()) == 1:
            batch_label = self.weight_ce[batch_label]  
        
        S_pp = in_package['S_pp']  
        
        Labels = batch_label
            
        if is_bias:
            S_pp = S_pp - self.vec_bias
        
        if not is_conservative:  
            S_pp = S_pp[:,self.seenclass]  
            Labels = Labels[:,self.seenclass]  
            assert S_pp.size(1) == len(self.seenclass)  
        
        Prob = self.log_softmax_func(S_pp)  
          
        loss = -torch.einsum('bk,bk->b',Prob,Labels)  
        loss = torch.mean(loss)  
        return loss  
    
    def compute_relaxed_cross_entropy(self,in_package):  
        assert self.is_bias == False
        assert self.is_conservative == True
        
        batch_label = in_package['batch_label'] 
        
        if len(in_package['batch_label'].size()) == 1:
            batch_label = self.weight_ce[batch_label]  
        
        S_pp = in_package['S_pp']  
        
        Labels = batch_label
        
        S_pp_relaxed = S_pp + self.margin_CE
        
        Prob = self.log_softmax_func(S_pp_relaxed)  
          
        loss = -torch.einsum('bk,bk->b',Prob,Labels)  
        loss = torch.mean(loss)  
        return loss  
    
    def conpute_w2v_deviation(self):
        V_n = self.compute_V()
        loss = torch.norm(V_n-self.init_w2v_att,dim=1)
        loss = torch.mean(loss)
        return loss
    
    def compute_diversity(self,in_package):
        A = in_package['A']
        A_p = in_package['A_p']
        A_f = torch.einsum('bi,bir->bir',A_p,A)
        
        R = A_f.size(2)
        
        loss = torch.einsum('bir,bjr->b',A_f,A_f)/(R*R)
        loss = torch.mean(loss)
        
        return loss
        
    def compute_attr_att_entropy(self,in_package):
        A_p = in_package['A_p']                 #[bi]
        loss_entropy = torch.einsum('bi,i->b',A_p,self.att_entropy)
        loss_entropy = torch.mean(loss_entropy)/self.att_entropy.size(0)
        return loss_entropy
    
    def compute_loss(self,in_package):
        
        if len(in_package['batch_label'].size()) == 1:
            in_package['batch_label'] = self.weight_ce[in_package['batch_label']]  
        
        ## loss rank  
        if self.loss_type == 'CE':
            loss_CE = self.compute_aug_cross_entropy(in_package)
        elif self.loss_type == 'rank':
            loss_CE = self.compute_loss_rank(in_package)  
        else:
            raise Exception('Unknown loss type')
        
        ## loss Laplace  
        loss_pmp = self.compute_loss_Laplace(in_package)#self.compute_cross_entropy_seen_unseen(in_package)#  
        
        ## entropy attr attention
        loss_entropy_attr=self.compute_attr_att_entropy(in_package)
        
        ## entropy prediction  
        S_pp = in_package['S_pp']  
        A_p = in_package['A_p'] 
        entropy = self.compute_entropy_with_logits(S_pp)  
        entropy_A_p = self.compute_entropy(A_p) 
          
        ## pruning loss  
        loss_prune = self.compute_loss_pruning(in_package)  
        
        ## w2v deviation loss
        if self.init_w2v_att is not None:
            loss_w2v = self.conpute_w2v_deviation()
        else:
            loss_w2v = loss_CE.new_full((1,),-1)
        
        ## total loss  
        loss = loss_CE + self.lambda_1*loss_entropy_attr + self.lambda_2*loss_prune + self.lambda_3*loss_pmp  
          
        out_package = {'loss':loss,'loss_CE':loss_CE,
                       'loss_entropy_attr':loss_entropy_attr,
                       'loss_w2v':loss_w2v,'loss_prune':loss_prune,
                       'loss_pmp':loss_pmp,
                       'entropy':entropy,'entropy_A_p':entropy_A_p}  
          
        return out_package  
    
    
        
    def extract_attention(self,imgs):
        Fs = self.base_ResNet(imgs)
        if self.is_conv:
            Fs = self.conv1(Fs)
            Fs = self.conv1_bn(Fs)
            Fs = F.relu(Fs)
        
        shape = Fs.shape
        Fs = Fs.reshape(shape[0],shape[1],shape[2]*shape[3])
        
        V_n = self.compute_V()
          
        if self.normalize_F and not self.is_conv:  
            Fs = F.normalize(Fs,dim = 1)
        
        A = torch.einsum('iv,vf,bfr->bir',V_n,self.W_2,Fs)   
        A = F.softmax(A,dim = -1)
        Hs = torch.einsum('bir,bfr->bif',A,Fs)
        
        package = {'A':A,'Hs':Hs}
        #What the attribute does not appear in the image
        return package        #bif
    
    def compute_attribute_embed(self,Hs):
        B = Hs.size(0)  
        V_n = self.compute_V()
        S_p = torch.einsum('iv,vf,bif->bi',V_n,self.W_1,Hs) 
        
        ## Attribute attention
        A_p = torch.einsum('iv,vf,bif->bi',V_n,self.W_3,Hs)
        A_p = torch.sigmoid(A_p) 
        ##  
        
        A_b_p = self.att.new_full((B,self.dim_att),fill_value = 1)  
        
        if self.uniform_att_2:
            S_pp = torch.einsum('ki,bi,bi->bik',self.att,A_b_p,S_p)
        else:
            S_pp = torch.einsum('ki,bi,bi->bik',self.att,A_p,S_p)
        
        if self.non_linear_emb:
            S_pp = torch.transpose(S_pp,2,1)    #[bki] <== [bik]
            S_pp = self.emb_func(S_pp)          #[bk1] <== [bki]
            S_pp = S_pp[:,:,0]                  #[bk] <== [bk1]
        else:
            S_pp = torch.sum(S_pp,axis=1)        #[bk] <== [bik]
        
        if self.is_bias:
            self.vec_bias = self.mask_bias*self.bias
            S_pp = S_pp + self.vec_bias
        
        package = {'S_pp':S_pp,'A_p':A_p}  
        
        return package  
    
    def forward(self,imgs):
        
        package_1 = self.extract_attention(imgs)
        Hs = package_1['Hs']
        package_2 = self.compute_attribute_embed(Hs)
        
        package_out = {'A':package_1['A'],'A_p':package_2['A_p'],'S_pp':package_2['S_pp']} 
        
        return package_out
    
