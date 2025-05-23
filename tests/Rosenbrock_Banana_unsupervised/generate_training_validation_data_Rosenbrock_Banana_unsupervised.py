################################################################################
# Read the pkl file with the MCMC chain and split the chain into training data #
# and validation data. The training data is used to train the model and the    #
# validation data is used to validate the model. Both are saved in a pkl file. #
################################################################################
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

# fix the seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)

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


def rosenbrock_log_pdf(x, c1=1.0, c2=20.0):
    """
    Compute the log PDF of the Rosenbrock distribution.
    The Rosenbrock distribution is defined as:
    f(x1, x2) = exp(-((x1 - c1)^2 + c2 * (x1^2 - x2)^2))
    where c1 and c2 are constants.

    Parameters
    ----------
    x : array-like
        The input point (x1, x2) for which to compute the log PDF.
    c1 : float
        Parameter for the Rosenbrock distribution.
    c2 : float
        Parameter for the Rosenbrock distribution.

    Returns
    -------
    log_pdf : float
        The log PDF of the Rosenbrock distribution at the input point (x1, x2).
    """
    x1, x2 = x
    return -((x1 - c1)**2 + c2 * (x1**2 - x2)**2)

def metropolis_hastings(log_pdf, initial, n_samples, proposal_std=0.5, burn_in=1000, c1=1.0, c2=20.0):
    """
    Metropolis-Hastings algorithm to sample from a distribution defined by log_pdf.
    
    Parameters
    ----------
    log_pdf : function
        Function that computes the log PDF of the target distribution.
    initial : array-like
        Initial point of the Markov chain.
    n_samples : int
        Number of samples to generate.
    proposal_std : float
        Standard deviation of the proposal distribution.
    burn_in : int
        Number of samples to discard as burn-in.
    c1 : float
        Parameter for the Rosenbrock distribution.
    c2 : float
        Parameter for the Rosenbrock distribution.

    Returns
    -------
    samples : array-like
        Generated samples from the target distribution.
    """
    samples = []
    x = np.array(initial)

    for i in range(n_samples + burn_in):
        # Propose a new point
        proposal = x + np.random.normal(scale=proposal_std, size=2)
        
        # Acceptance probability
        log_accept_ratio = log_pdf(proposal, c1, c2) - log_pdf(x, c1, c2)
        if np.log(np.random.rand()) < log_accept_ratio:
            x = proposal

        if i >= burn_in:
            samples.append(x.copy())

    return np.array(samples)

def generate_rosenbrock_data(n_samples, c1=1.0, c2=20.0):
    """
    Generate samples from the Rosenbrock distribution using the Metropolis-Hastings algorithm.
    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    c1 : float
        Parameter for the Rosenbrock distribution.
    c2 : float
        Parameter for the Rosenbrock distribution.

    Returns
    -------
    samples : array-like
        Generated samples from the Rosenbrock distribution.
    """
    initial = [0.0, 0.0]  # Starting point of the chain
    return metropolis_hastings(rosenbrock_log_pdf, initial, n_samples, c1=c1, c2=c2)


if __name__ == "__main__":
    # Define the path to save the training and validation data
    PATH_pklfile_training_data = 'training_data_Rosenbrock_Banana_unsupervised.pkl'
    PATH_pklfile_validation_data = 'validation_data_Rosenbrock_Banana_unsupervised.pkl'

    # Generate training data with the Rosenbrock-banana distribution
    fraction_of_validation_data = 0.2
    n_rosenbrock_samples = 100000
    sigma_1 = 1.0
    sigma_2 = 20.0
    chain = generate_rosenbrock_data(n_rosenbrock_samples)

    # Save the training and validation data
    training_data, validation_data = split_chain_into_training_and_validation_data(chain, fraction_of_validation_data)
    save_training_and_validation_data(training_data, validation_data, PATH_pklfile_training_data, PATH_pklfile_validation_data)
    print(f"Shape of the training data: {training_data.shape}")
    print(f"Shape of the validation data: {validation_data.shape}")
    print(f"Training and validation data saved in {PATH_pklfile_training_data} and {PATH_pklfile_validation_data}")

    # Visualize the generated samples
    plt.scatter(chain[:, 0], chain[:, 1], s=1, alpha=0.3)
    plt.title("Rosenbrock Banana Distribution Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()
