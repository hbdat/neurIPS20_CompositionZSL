# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 23:02:21 2019

@author: Warmachine
"""

import os,sys
pwd = os.getcwd()
sys.path.insert(0,pwd) 
print('-'*30)
print(os.getcwd())
print('-'*30)
#%%
import tensorflow as tf
import numpy as np
import pdb
from global_setting_Pegasus import NFS_path
import pickle
idx_GPU=0
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(idx_GPU)
# Seen and unseen here is only used for zeroshot experiment
#%%
dataset = 'CUB'
context_size = 10
model_name = 'glove-wiki-gigaword-300'
wiki_context = './w2v/{}_contexts_{}_{}.pkl'.format(dataset,context_size,model_name)

save_path = './w2v/{}_class.pkl'.format(dataset)

dim_w2v = 300
learning_rate = 1
n_iters = 100001
k=5
batch_size=-1
is_save = False
decay = 1
#%% load context information
with open(wiki_context,'rb') as infile:
    contexts,classes = pickle.load(infile)
classes = np.array(classes)
n_classes = len(contexts)
#%%
input_contexts = tf.placeholder(shape=[None,dim_w2v],dtype=tf.float32,name='input_contexts')
embedding = tf.get_variable(name='embedding',shape=[dim_w2v,n_classes],dtype=tf.float32)
logits=tf.matmul(input_contexts,embedding)
labels = tf.placeholder(shape=[None,n_classes],dtype=tf.float32,name='labels')
loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,logits=logits)

n_embedding=tf.nn.l2_normalize(embedding,axis=0)
similarity = tf.matmul(tf.transpose(n_embedding),n_embedding)
top_k_similar=tf.nn.top_k(similarity,k)

#%%
lr = tf.Variable(learning_rate)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
grads = optimizer.compute_gradients(loss)
train = optimizer.apply_gradients(grads)
update_lr = tf.assign(lr,lr*decay)
#%%
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
#%%
#pdb.set_trace()
print('-'*30)
print('embedding')
iters = n_iters
for i in range(iters):
    idx_class = i%n_classes
    if contexts[idx_class] is None:
        continue
    tr_contexts = contexts[idx_class]
    mask_nan = ~ np.isnan(np.sum(tr_contexts,axis=1))
    
    tr_contexts = tr_contexts[mask_nan]
    
    tr_labels = np.zeros((tr_contexts.shape[0],n_classes))#tf.one_hot([idx_class]*batch_size,n_labels)
    tr_labels[:,idx_class]=1.0
    
    _,l_v=sess.run([train,loss],feed_dict={input_contexts:tr_contexts,labels:tr_labels})
#    print(l_v)
    if i%1000==0:
        print('-'*10)
        print(i,iters)
        value_similar,idx_similar=sess.run(top_k_similar)
        
        for idx_class in range(0,10):
            str_display = ''
#            pdb.set_trace()
            similar_concepts=classes[idx_similar[idx_class]]
            for concept in similar_concepts: str_display+= concept +'|'
            print(str_display)
        print(l_v)
        print('lr {}'.format(update_lr.eval()))
#%%
# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(200, 200))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(
        label,
        xy=(x, y),
        xytext=(5, 2),
        textcoords='offset points',
        ha='right',
        va='bottom')

  plt.savefig(filename)

final_embeddings = tf.transpose(n_embedding).eval()

with open(save_path,'wb') as f:
    pickle.dump(final_embeddings,f)
try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(
      perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  low_dim_embs = tsne.fit_transform(final_embeddings)#[:plot_only, :]
  
#  pdb.set_trace()
  plot_with_labels(low_dim_embs, classes,'./plots/w2v_{}_contexts_{}_{}.png'.format(dataset,context_size,model_name))

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)
