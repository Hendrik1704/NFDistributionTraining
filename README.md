# NFDistributionTraining

This repository contains the code for training a normalizing flow (NF) model
on a dataset containing of an array of many samples (rows) and some features (columns).
The last column of the dataset contains the log-likelihood of the training distribution for 
the given parameter (supervised learning).
In the case where the log-likelihood is not available, the code can be adapted to
use another loss function to train the NF model (unsupervised learning).

The NF code and model is adapted from [arXiv:2310.04635](https://arxiv.org/pdf/2310.04635) 
and the code is a rewritten and extended version from this [GitLab](https://gitlab.com/yyamauchi/rbm_nf/-/tree/main) repository.

## Example Usage

### Supervised Learning

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

### Unsupervised Learning
To run the unsupervised learning example, go to the `tests/Rosenbrock_Banana_unsupervised` 
directory and run the following script:
```bash
python3 generate_training_validation_data_Rosenbrock_Banana_unsupervised.py
```
This will generate the training and validation data for the Rosenbrock Banana distribution
without the log-likelihood column.
The default generation will create 100000 samples from the distribution and saves
20% of the samples for validation.
It will produce files called `training_data_Rosenbrock_Banana_unsupervised.pkl` and
`validation_data_Rosenbrock_Banana_unsupervised.pkl` in the
`tests/Rosenbrock_Banana_unsupervised` directory.

Then run the following script to train the NF model:
```bash
python3 Rosenbrock_Banana_NF_unsupervised.py
```
This will load the training data, train the NF model, and save the model to a file
called `normalizing_flow_model_Rosenbrock_Banana_unsupervised.pkl` in the
`tests/Rosenbrock_Banana_unsupervised` directory.
It will also plot the training versus the NF model samples from the distribution.
The unsupervised learning example uses the same NF model as the supervised learning example,
but the training is done using a different loss function.

## Custom prior class for pocoMC
If you want to use the NF model as a prior for pocoMC, you can create a class 
similar to the one in the `tests/custom_prior_distribution_class/custom_prior.py` 
file. There is an example of how to use the NF model as a prior similar to the 
custom implementation in the pocoMC [examples](https://pocomc.readthedocs.io/en/latest/priors.html).
In this example and test of the implementation the Rosenbrock Banana distribution is used.

## Further examples and tests
Other examples and tests are provided in the `tests` directory (see `README.md`).

## Requirements
A `requirements.txt` file is provided to install the required packages.
This is just a list of packages with which the code has been tested.
Lower or upper versions of the packages may work, but are not guaranteed to work.
The code is tested with Python 3.10.17 and the list of packages provided.