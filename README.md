# NFDistributionTraining

This repository contains the code for training a normalizing flow (NF) model
on a dataser containing of an array of many samples (rows) and some features (columns).
The last column of the dataset contains the log-likelihood of the training distribution for 
the given parameter.

The NF code and model is adapted from [arXiv:2310.04635](https://arxiv.org/pdf/2310.04635) 
and the code is a rewritten version from this [GitLab](https://gitlab.com/yyamauchi/rbm_nf/-/tree/main) repository.