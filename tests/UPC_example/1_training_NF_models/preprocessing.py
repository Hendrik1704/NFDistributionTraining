import pickle
import numpy as np
import random
import os

def split_chain_into_training_and_validation_data(chain, fraction_of_validation_data):
    """
    Split the chain into training and validation data.
    Choose the indices of the validation data randomly.
    The training data is the rest of the chain.

    Parameters
    ----------
    chain : array-like
        The chain to be split into training and validation data.
    fraction_of_validation_data : float
        The fraction of the chain to be used for validation data.
        The rest will be used for training data.

    Returns
    -------
    training_data : array-like
        The training data.
    validation_data : array-like
        The validation data.
    """
    n_samples = chain.shape[0]
    n_validation_samples = int(n_samples * fraction_of_validation_data)
    indices = list(range(n_samples))
    validation_indices = random.sample(indices, n_validation_samples)
    training_indices = list(set(indices) - set(validation_indices))
    training_data = chain[training_indices]
    validation_data = chain[validation_indices]
    return training_data, validation_data

def save_training_and_validation_data(training_data, validation_data, PATH_pklfile_training_data, PATH_pklfile_validation_data):
    """
    Save the training and validation data in pkl files.

    Parameters
    ----------
    training_data : array-like
        The training data to be saved.
    validation_data : array-like
        The validation data to be saved.
    PATH_pklfile_training_data : str
        The path to the pkl file where the training data will be saved.
    PATH_pklfile_validation_data : str
        The path to the pkl file where the validation data will be saved.
    """
    with open(PATH_pklfile_training_data, 'wb') as pf:
        pickle.dump(training_data, pf)
    with open(PATH_pklfile_validation_data, 'wb') as pf:
        pickle.dump(validation_data, pf)

def read_pkl_file_chain(PATH_pklfile_chain):
    """
    Data is a dictionary containing:
    - 'chain'
    - 'weights'
    - 'logl'
    - 'logp'
    - 'logz'
    - 'logz_err'
    as keys. Only use 'chain' here.
    """
    with open(PATH_pklfile_chain, 'rb') as pf:
        data = pickle.load(pf)

    return data['chain'], data['logl']

seed = 42
np.random.seed(seed)
random.seed(seed)

chain, logl = read_pkl_file_chain('./mcmc_PCGP_eP_vanilla/chain.pkl')
chain = np.hstack((chain, logl.reshape(-1, 1)))
chain = np.delete(chain, [3, 8, 9, 10], axis=1)

training_data, validation_data = split_chain_into_training_and_validation_data(chain, 0.2)

os.makedirs("data_eP", exist_ok=True)
save_training_and_validation_data(training_data, validation_data, "data_eP/training_data.pkl", "data_eP/validation_data.pkl")
