################################################################################
# Custom Prior Distribution Class Example - Use with pocoMC                    #
################################################################################
import sys
import os

# Get the path two levels up and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import optax
import jax
import jax.numpy as jnp
import time
import pickle
from functools import partial
import src.train_nf as train_nf
from src.realnvp.utils import load
import numpy as np
import corner
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import simpson

class CustomPrior:
    def __init__(self, path_NF_model, seed=None):
        self.dim = 2
        self.bounds = np.array([[-2.0, 3.0],
                                [-1.0, 8.0]])
        
        # Specify to use CPU, not GPU.
        jax.config.update('jax_platform_name', 'cpu')

        if seed is None:
            seed = time.time_ns()
        self.sample_key, self.init_key = jax.random.split(jax.random.PRNGKey(seed), 2)

        # Load the normalizing flow and its hyperparameters
        self.flow, self.hyperparams = load(path_NF_model, self.init_key)
        self.dimension = self.hyperparams['dimension']

    def logpdf(self, x):
        """
        Compute the log probability density function of the custom prior distribution using the log_probab() method of the normalizing flow.
        """
        log_probab = self.flow.log_prob(x)
        return log_probab

    def rvs(self, size=1):
        self.sample_key, pkey = jax.random.split(self.sample_key)
        z = jax.random.normal(pkey, (size, self.dimension))
        x, _ = self.flow(z)
        return x

def train_NF_model():
    training_data_path = "training_data_Rosenbrock_Banana.pkl"
    dimension = 2
    normalizing_flow_model_path = "normalizing_flow_model_Rosenbrock_Banana.pkl"
    test_data_path = "validation_data_Rosenbrock_Banana.pkl"
    batch_size_training = 4000
    layers = 6
    learning_rate = 1e-3
    number_training_steps = 7500
    seed = 42
    use_KL_divergence_loss = True
    initial_normalizing_flow_model_path = None
    loss_threshold_early_stopping = 0
    optimizer = optax.adam
    print_loss_training = True
    metrics_mid_training = False
    metrics_mid_training_frequency = 1000
    train_nf.train(training_data_path, dimension, normalizing_flow_model_path, 
          test_data_path, batch_size_training, layers, learning_rate, 
          number_training_steps, seed, use_KL_divergence_loss, 
          initial_normalizing_flow_model_path, loss_threshold_early_stopping,
          optimizer, print_loss_training, metrics_mid_training,
          metrics_mid_training_frequency)

def load_comparison_dataset(path_comparison_data):
    with open(path_comparison_data, 'rb') as pf:
        data_raw = pickle.load(pf)
    return np.array(data_raw[:, :-1])

def plot_comparison_prior_samples_to_training_data(samples, path_comparison_data, plot_filename):
    data_theta = load_comparison_dataset(path_comparison_data)
    data_length = len(data_theta)
    data_ratio = len(samples) / data_length
    binwidth = 0.1  # Width of the bins for the histogram

    figure1 = corner.corner(
        data_theta,
        weights=np.ones(len(data_theta)) * data_ratio,
        labels=[f'$x_{i}$' for i in range(2)],
        labelpad=0.2,
        bins=[int((max(data_theta[:, i]) - min(data_theta[:, i])) // binwidth) for i in range(2)],
        color='black',
        label_kwargs={'fontsize': 20},
        hist_kwargs={"linewidth": 2},
        quantiles=None,
        smooth=(1.7),
        smooth1d=1.0,
        show_titles=True,
    )

    figure2 = corner.corner(
        samples,
        fig=figure1,
        bins=[int((max(samples[:, i]) - min(samples[:, i])) / binwidth) for i in range(2)],
        color='green',
        labels=[f'$x_{i}$' for i in range(2)],
        labelpad=0.2,
        label_kwargs={'fontsize': 20},
        hist_kwargs={"linewidth": 2},
        quantiles=None,
        smooth=(1.7),
        smooth1d=1.0,
        show_titles=True,
    )

    width = 8
    height = 6
    figure2.set_size_inches(width, height)

    plt.legend(['Training Data', 'Normalizing Flow Prior'], loc='upper right', fontsize=10)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    if plot_filename is not None:
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.clf()

def plot_samples(samples, plot_filename=None):
    plt.figure(figsize=(8, 6))
    plt.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5)
    plt.title("Samples from Custom Prior Distribution")
    plt.xlabel("x0")
    plt.ylabel("x1")
    if plot_filename is not None:
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.clf()

def plot_logpdf(samples, custom_prior, plot_filename=None):
    x0_min, x0_max = np.percentile(samples[:, 0], [0.1, 99.9])
    x1_min, x1_max = np.percentile(samples[:, 1], [0.1, 99.9])
    x0 = np.linspace(x0_min, x0_max, 500)
    x1 = np.linspace(x1_min, x1_max, 500)
    xx0, xx1 = np.meshgrid(x0, x1)
    grid = np.stack([xx0.ravel(), xx1.ravel()], axis=1)

    logpdf_vals = custom_prior.logpdf(jnp.array(grid))
    pdf_vals = np.exp(logpdf_vals).reshape(xx0.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx0, xx1, pdf_vals, levels=100, cmap='Purples')
    plt.title("NF PDF from logpdf")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.colorbar(label="Density")
    if plot_filename is not None:
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.clf()

    # Normalize integral over this domain
    integral = simpson(simpson(pdf_vals, x1), x0)
    print(f"Integral over pdf domain: {integral:.4f}")

def timing_sample_and_logpdf(custom_prior, samples):
    t0 = time.time()
    _ = custom_prior.rvs(size=50000)
    print("Sampling time:", time.time() - t0)

    t0 = time.time()
    _ = custom_prior.logpdf(jnp.array(samples))
    print("logpdf time:", time.time() - t0)

if __name__ == "__main__":
    # Train the normalizing flow model after running generate_training_validation_data_Rosenbrock_Banana.py
    # This script assumes that the training and validation data has already been generated.
    # If not, run generate_training_validation_data_Rosenbrock_Banana.py first.
    train_NF_model()

    # Create an instance of the custom prior distribution class
    path_NF_model = "normalizing_flow_model_Rosenbrock_Banana.pkl"
    custom_prior = CustomPrior(path_NF_model)
    # Generate samples from the custom prior distribution
    samples = custom_prior.rvs(size=50000)
    samples = np.array(samples)
    
    plot_comparison_prior_samples_to_training_data(
        samples, 
        "training_data_Rosenbrock_Banana.pkl", 
        "corner_plot_custom_prior_samples.png"
    )
    plot_samples(samples, "scatter_plot_custom_prior_samples.png")
    plot_logpdf(samples, custom_prior, "logpdf_custom_prior_samples.png")
    timing_sample_and_logpdf(custom_prior, samples)
