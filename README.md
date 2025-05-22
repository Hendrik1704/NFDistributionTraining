# NFDistributionTraining

This repository contains the code for training a normalizing flow (NF) model
on a dataset containing of an array of many samples (rows) and some features (columns).
The last column of the dataset contains the log-likelihood of the training distribution for 
the given parameter.

The NF code and model is adapted from [arXiv:2310.04635](https://arxiv.org/pdf/2310.04635) 
and the code is a rewritten version from this [GitLab](https://gitlab.com/yyamauchi/rbm_nf/-/tree/main) repository.


## Example Usage

Go to the `tests/Rosenbrock_Banana` directory and run the following scripts:

```bash
python3 generate_training_validation_data_Rosenbrock_Banana.py
```
To generate the training and validation data for the Rosenbrock Banana distribution.
The default generation will create 100000 samples from the distribution and saves
20% of the samples for validation.
It will produce files called `training_data_Rosenbrock_Banana.pkl` and
`validation_data_Rosenbrock_Banana.pkl` in the `tests/Rosenbrock_Banana` directory.

Then run the following script to train the NF model:
```bash
python3 Rosenbrock_Banana_NF.py
```
This will load the training data, train the NF model, and save the model to a file 
called `normalizing_flow_model_Rosenbrock_Banana.pkl` in the 
`tests/Rosenbrock_Banana` directory.
It will also save some metrics to a file called `metrics_Rosenbrock_Banana.dat` 
in the same directory and generate a plot of the training versus the NF model 
samples from the distribution.

## Requirements
A `requirements.txt` file is provided to install the required packages.
This is just a list of packages with which the code has been tested.
Lower or upper versions of the packages may work, but are not guaranteed to work.
The code is tested with Python 3.10.17 and the list of packages provided.