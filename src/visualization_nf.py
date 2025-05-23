#!/usr/bin/env python

import jax
import jax.numpy as jnp
import time
from functools import partial
import pickle
import corner
import numpy as np
import matplotlib.pyplot as plt
from .realnvp.utils import load

def visualize(path_normalizing_flow, path_comparison_data, number_samples,
              plot_filename=None, seed=None, binwidth=0.04, 
              unsupervised_training=False):
    """
    Samples from a trained normalizing flow and generates a corner plot to 
    compare to other data.

    Parameters
    ----------
    path_normalizing_flow : str
        Path to the trained normalizing flow.
    path_comparison_data : str
        Path to the data to compare to.
    number_samples : int
        Number of samples to generate.
    plot_filename : str, optional
        Name of the file to save the plot to. If None, the plot will not 
        be saved.
    seed : int, optional
        Seed for the random number generator. If None, a random seed 
        will be used.
    binwidth : float, optional
        Width of the bins for the histogram. Defaults to 0.04.
    unsupervised_training : bool, optional
        If True, the normalizing flow is trained in an unsupervised 
        manner. Defaults to False.
    """
    # Specify to use CPU, not GPU.
    jax.config.update('jax_platform_name', 'cpu')

    if seed is None:
        seed = time.time_ns()
    sample_key, init_key = jax.random.split(jax.random.PRNGKey(seed), 2)

    # Load the normalizing flow and its hyperparameters
    flow, hyperparams = load(path_normalizing_flow, init_key)
    dimension = hyperparams['dimension']

    @partial(jnp.vectorize, signature='(i)->(i)')
    def map_function(x):
        y, log_det = flow(x)
        return y

    @jax.jit
    def sample(key):
        x = jax.random.normal(key, (number_samples, dimension))
        y = map_function(x)
        return y

    observations = sample(sample_key)
    observations = np.array(observations)
    with open(path_comparison_data, 'rb') as pf:
        data_raw = pickle.load(pf)
    if not unsupervised_training:
        data_theta = np.array(data_raw[:, :-1])
    else:
        data_theta = np.array(data_raw)
    data_length = len(data_theta)
    data_ratio = number_samples / data_length

    figure1 = corner.corner(
        data_theta,
        weights=np.ones(len(data_theta)) * data_ratio,
        labels=[f'$x_{i}$' for i in range(dimension)],
        labelpad=0.2,
        bins=[int((max(data_theta[:, i]) - min(data_theta[:, i])) // binwidth) for i in range(dimension)],
        color='black',
        label_kwargs={'fontsize': 20},
        hist_kwargs={"linewidth": 2},
        quantiles=None,
        smooth=(1.7),
        smooth1d=1.0,
        show_titles=True,
    )

    figure2 = corner.corner(
        observations,
        fig=figure1,
        bins=[int((max(observations[:, i]) - min(observations[:, i])) / binwidth) for i in range(dimension)],
        color='green',
        labels=[f'$x_{i}$' for i in range(dimension)],
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

    plt.legend(['Training Data', 'Normalizing Flow'], loc='upper right', fontsize=10)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    if plot_filename is not None:
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
    else:
        plt.show()