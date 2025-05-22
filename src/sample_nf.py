#!/usr/bin/env python

from functools import partial
import time
import jax
import jax.numpy as jnp

from .realnvp.utils import load

def sample(
    path_normalizing_flow,
    number_samples,
    seed=None,
    print_samples=False
):
    """
    Samples from a trained normalizing flow.

    Parameters
    ----------
    path_normalizing_flow : str
        Path to the trained normalizing flow.
    number_samples : int
        Number of samples to generate.
    seed : int, optional
        Seed for the random number generator. If None, a random seed will be used.
    print_samples : bool, optional
        Flag indicating whether to print the samples or return them. Defaults to False.

    Returns
    -------
    jnp.ndarray or None
        Generated samples if print_samples is False, otherwise None.
    """
    # Specify to use CPU, not GPU.
    jax.config.update('jax_platform_name', 'cpu')

    if seed is None:
        seed = time.time_ns()
    sample_key, init_key = jax.random.split(jax.random.PRNGKey(seed), 2)

    # Load the normalizing flow and its hyperparameters
    flow, hyperparams = load(path_normalizing_flow, init_key)
    dimension = hyperparams['dimension']

    @partial(jnp.vectorize, signature='(i)->(i)', excluded={1})
    def map_function(x, flow):
        y, log_det = flow(x)
        return y

    sample_key, pkey = jax.random.split(sample_key)
    x = jax.random.normal(pkey, (number_samples, dimension))
    if print_samples:
        y = map_function(x, flow)
        for i in range(number_samples):
            print(*y[i])
    else:
        return map_function(x, flow)