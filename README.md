# Compositional Zero-Shot Learning via Fine-Grained Dense Feature Composition

## Overview
This repository contains the implementation of [Compositional Zero-Shot Learning via Fine-Grained Dense Feature Composition](https://hbdat.github.io/pubs/neurips20_CompositionZSL_final.pdf).
> In this work, we transform a discriminative model into a generative model to improve zero-shot learning by leveraging compositionality of dense representations based on [Dense-Attention Zero-shot Learning (DAZLE)](https://github.com/hbdat/cvpr20_DAZLE)

![Image](https://github.com/hbdat/neurIPS20_CompositionZSL/raw/main/fig/feature_composition.png)

## Notes
```
The repository is still under construction.
Please let me know if you encounter any issues.

Best,
Dat Huynh

huynh [dot] dat [at] northeastern [dot] edu
```

---
## Prerequisites
To install all the dependency packages, please run:
```
pip install -r requirements.txt
```

---
## Data Preparation
1) Please download and extract information into the `./data folder`. We include details about download links as well as what are they used for in each folder within `./data folder`.

2) **[Optional]** For DeepFashion dataset, we partition seen/unseen classes and training/testing split via:
```
python ./extract_feature/extract_annotation_DeepFashion.py							#create ./data/DeepFashion/annotation.pkl
```
We have included the result file by default in the repository. Similarly, we have also included the attribute semantics from GloVe model for all datasets which are computed by:
```
python ./extract_feature/extract_attribute_w2v_DeepFashion.py						        #create ./w2v/DeepFashion_attribute.pkl
python ./extract_feature/extract_attribute_w2v_AWA2.py								#create ./w2v/AWA2_attribute.pkl
python ./extract_feature/extract_attribute_w2v_CUB.py								#create ./w2v/CUB_attribute.pkl
python ./extract_feature/extract_attribute_w2v_SUN.py								#create ./w2v/SUN_attribute.pkl
```

3) Please run feature extraction scripts in `./extract_feature` folder to extract features from the last convolution layers of ResNet as region features for attention mechanism:
```
python ./extract_feature/extract_feature_map_ResNet_101_DeepFashion.py				        #create ./data/DeepFashion/feature_map_ResNet_101_DeepFashion_sep_seen_samples.hdf5
python ./extract_feature/extract_feature_map_ResNet_101_AWA2.py						#create ./data/AWA2/feature_map_ResNet_101_AWA2.hdf5
python ./extract_feature/extract_feature_map_ResNet_101_CUB.py						#create ./data/CUB/feature_map_ResNet_101_CUB.hdf5
python ./extract_feature/extract_feature_map_ResNet_101_SUN.py						#create ./data/SUN/feature_map_ResNet_101_SUN.hdf5
```
These scripts create hdf5 files which contain image features and data splits for training and evaluation.

---
## Pre-trained Setting
1) We provide separate jupyter notebooks for training and evaluation on all four datasets in `./notebook`  folder:
```
./notebook/Composer/Composer_DeepFashion.ipynb
./notebook/Composer/Composer_AWA2.ipynb
./notebook/Composer/Composer_CUB.ipynb
./notebook/Composer/Composer_SUN.ipynb
```

## Fine-tune Setting
Coming soon ...

---
## Citation
If you find the project helpful, we would appreciate if you cite the works:
```
@article{Huynh-Composer:NeurIPS20,
  author = {D.~Huynh and E.~Elhamifar},
  title = {Compositional Zero-Shot Learning via Fine-Grained Dense Feature Composition},
  journal = {Conference on Neural Information Processing Systems},
  year = {2020}}
```

```
@article{Huynh-DAZLE:CVPR20,
  author = {D.~Huynh and E.~Elhamifar},
  title = {Fine-Grained Generalized Zero-Shot Learning via Dense Attribute-Based Attention},
  journal = {{IEEE} Conference on Computer Vision and Pattern Recognition},
  year = {2020}}
```

---
## References
We adapt our dataloader modules from:
https://github.com/edgarschnfld/CADA-VAE-PyTorch
and Nonnegative OMP from:
https://github.com/davebiagioni/pyomp
