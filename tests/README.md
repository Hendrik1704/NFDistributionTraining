## Example Usage

This README file provides instructions on how to use the code in the `tests` 
directory for the tests that are not described in the main README file already.

### Rosenbrock Banana Continue Training File
This example shows how to continue training a NF model from a saved model file.
First, generate the training and validation data by running the following script:
```bash
python3 generate_training_validation_data_Rosenbrock_Banana.py
```
This will create the training and validation data files. Then, run the 
first training script, that will be interrupted after a set time:
```bash
python3 Rosenbrock_Banana_NF.py
```
This will start the training of the NF model and after a set time (45 seconds by default)
it will simulate a Ctrl+C interrupt to stop the training. The NF model state will be saved
to a file called `normalizing_flow_model_Rosenbrock_Banana.pkl` in the 
`tests/Rosenbrock_Banana_Continue_Training_File` directory.

Finally, run the second training script to continue the training from the saved model file:
```bash
python3 Rosenbrock_Banana_NF_Continue_From_File.py
```
This will load the saved NF model and continue the training for another set number of steps.

## UPC example
This example uses a real physics dataset from ultra peripheral collisions (UPC) 
of lead ions. The dataset is taken from [Zenodo](https://zenodo.org/records/15880667) 
`Global Bayesian Analysis of J/Psi Photoproduction on Proton and Lead Targets`.
We use the MCMC chain file for the $\gamma+p$ data without the $K$-factor, to
train a normalizing flow model to learn the distribution of the data.
The first step is to separate the chain into training and validation data. This can be done
by running the following script in the `UPC_example/1_training_NF_models` directory:
```bash
python3 preprocessing.py
```
This will create the training and validation data files. Then, run the 
training script to train the NF model in different configurations. This can be
done by executing the `train_nf_job.py` script with the index of the configuration
you want to run. The slurm job script `submit_jobs.slurm` can be used (probably
with small cluster specific modifications) to submit the jobs.
In this example 80 different configurations are tested, with different NF architectures
and hyperparameters. The results of the training (trained models) are saved in the
`models_eP` directory.
The jupyter notebook `FindOptimalNFModel.ipynb` can then be used to find the best
model based on the comparison of the NF samples to the training data using the KL divergence.
This notebook can also be used to create plots comparing the best NF model to the training data.

For the MCMC step, we only give some dummy scripts that show how to use the trained NF model
in an MCMC run. The actual MCMC script highly depends on the specific use case and is not
part of this repository. The scripts `MCMC_CustomPrior_eP_vanilla.py` and the notebook `PlotsMCMC.ipynb`
are just examples that need to be adapted to the specific use case.