# Benchmarking Deep Learning-based Models on Nanophotonic Inverse Design Problems

This repository is the implementation of paper [Benchmarking deep learning-based models on nanophotonic inverse design problems](https://www.oejournal.org/article/doi/10.29026/oes.2022.210012)

# Introduction
## 1. Two inverse design tasks:
![Geometry illustration](./Figures/Structures.png)
### (a) Template structure: Silicon Nanorods for structural color inverse design. 

This structure can be described by 4 parameters: Period (P), Diameter (D), Gap (G), and Height (H). 

The optical response is reflection structural color. 

### (b) Freeform structure: Transmission spectrum inverse design based on the Si free-from structures. 

This structure is described by a pixlated image. 

The optical response is the transmission. 

## 2. Three examined deep learning models
![Model illustrations](./Figures/Models.png)
### (b) [Tandem networks](https://onlinelibrary.wiley.com/doi/10.1002/adma.201905467)
### (c) Variational Auto-Encoders [(VAE)](https://proceedings.neurips.cc/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf)
### (d) Generative Adversarial Networks [(GAN)](https://arxiv.org/pdf/1411.1784.pdf)

## 3. Three evaluation metrics
![Metrics summary](./Figures/Metrics.png)



## How to load files



### 
Due to the file size limitation in Github, please download the training data, several trained models, and the predicted results here: . After download, add those folders under './task_2_free_form/....'.



(we are still construct this dataset and code base). 
