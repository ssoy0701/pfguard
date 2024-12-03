# PFGuard

#### Authors: Soyeon Kim, Yuji Roh, Geon Heo, and Steven Euijong Whang
<!-- #### In Proceedings of the 9th International Conference on Learning Representations (ICLR), 2021 -->

This repo contains codes used in the arXiv paper: [PFGuard: A Generative Framework with Privacy and Fairness Safeguards](https://arxiv.org/abs/2410.02246) 

*Abstract: Generative models must ensure both privacy and fairness for Trustworthy AI. While these goals have been pursued separately, recent studies propose to combine existing privacy and fairness techniques to achieve both goals. However, naively combining these techniques can be insufficient due to privacy-fairness conflicts, where a sample in a minority group may be amplified for fairness, only to be suppressed for privacy. We demonstrate how these conflicts lead to adverse effects, such as privacy violations and unexpected fairness-utility tradeoffs. To mitigate these risks, we propose PFGuard, a generative framework with privacy and fairness safeguards, which simultaneously addresses privacy, fairness, and utility. By using an ensemble of multiple teacher models, PFGuard balances privacy-fairness conflicts between fair and private training stages and achieves high utility based on ensemble learning. Extensive experiments show that PFGuard successfully generates synthetic data on high-dimensional data while providing both convergence in fair generative modeling and strict DP guarantees - the first of its kind to our knowledge.*

<!--
## Setting
This directory is for simulating FairBatch on the synthetic dataset.
The program needs PyTorch and Jupyter Notebook.

The directory contains a total of 4 files and 1 child directory: 
1 README, 2 python files, 1 jupyter notebook, 
and the child directory containing 6 numpy files for synthetic data.

## Simulation
To simulate FairBatch, please use the **jupyter notebook** in the directory.

The jupyter notebook will load the data and train the models with three 
different fairness metrics: equal opportunity, equalized odds, and demographic parity.

Each training utilizes the FairBatch sampler, which is defined in FairBatchSampler.py.
The pytorch dataloader serves the batches to the model via the FairBatch sampler. 
Experiments are repeated 10 times each.
After the training, the test accuracy and fairness will be shown.

## Other details
The two python files are models.py and FairBatchSampler.py.
The models.py file contains a logistic regression architecture and a test function.
The FairBatchSampler.py file contains two classes: CustomDataset and FairBatch. 
The CustomDataset class defines the dataset, and the FairBatch class implements 
the algorithm of FairBatch as described in the paper.

More detailed explanations of each component can be found in the code as comments.
Thanks!

## Demos using Google Colab
We also release Google Colab notebooks for fast demos.
You can access both the [PyTorch version](https://colab.research.google.com/drive/192tZmf-jXg1uesHW2TSqv0LoDbhAW4X1?usp=sharing) and the [TensorFlow version](https://colab.research.google.com/drive/1VBc7osg-wRKTKav32k1wY2yfKWK-wnDW?usp=sharing).

## Reference
```
@inproceedings{
roh2021fairbatch,
title={FairBatch: Batch Selection for Model Fairness},
author={Yuji Roh and Kangwook Lee and Steven Euijong Whang and Changho Suh},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=YNnpaAKeCfx}
}
```
-->
