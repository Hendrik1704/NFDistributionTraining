#!/usr/bin/env python

import time
import jax
import jax.numpy as jnp
import pickle
import numpy as np
import json
from functools import partial
from scipy.stats import kurtosis, skew

from .realnvp.utils import load

def metrics(normalizing_flow_model, training_data_path, metrics_path=None, 
            n_samples=None, metric_selection=[1, 1, 1, 1, 1], seed=None):
    """
    Compute and store metrics for a normalizing flow model during training.
    Evaluation of the trained model.

    Parameters
    ----------
    normalizing_flow_model_path : str or array
        Path to the normalizing flow model or an array type containing the 
        flow and hyperparameters.
    training_data_path : str
        Path to the training data file for comparison with the normalizing flow.
    metrics_path : str, optional
        Path to the file for storing metrics. If not provided, metrics
        will be returned as a list of strings.
    n_samples : int, optional
        Number of samples to draw from the normalizing flow for metric estimation.
        If None, the number of samples equals the size of the training dataset.
    metric_selection : list, optional
        List of 1s and 0s (or True and False) to select which metrics to compute.
        Considers the first 4 moments and covariances. Defaults to [1, 1, 1, 1, 1].
    seed : int, optional
        Random seed for sampling. If None, the current time in nanoseconds is used.
    """
    if seed is None:
        seed = time.time_ns()
    sample_key, initial_key = jax.random.split(jax.random.PRNGKey(seed), 2)

    # Load the training data
    with open(training_data_path, 'rb') as pf:
        training_data_raw = pickle.load(pf)
    training_data_theta = jnp.array(training_data_raw[:, :-1])
    training_data_distribution_value = jnp.array(training_data_raw[:, -1])
    n_samples_data = len(training_data_distribution_value)
    if n_samples is None:
        n_samples = n_samples_data

    # Load model
    if isinstance(normalizing_flow_model, str):
        try:
            flow, hyperparams = load(normalizing_flow_model, initial_key)
        except FileNotFoundError:
            print('Normalizing flow model path not found')
            exit(0)
    else:
        try:
            flow, hyperparams = normalizing_flow_model
        except ValueError:
            print('Model must be a path or a (flow, hyperparams) tuple')
            exit(0)

    n_dimensions = hyperparams['dimension']
    n_training = hyperparams['n_samples']
    n_layers = hyperparams['layers']
    n_batch_size_training = hyperparams['batch_size_training']
    n_training_steps = hyperparams['number_training_steps']

    if training_data_theta.shape[1] != n_dimensions:
        print('Dimension mismatch between training data and model')
        exit(0)

    # Create subsets for uncertainty estimation
    if n_samples_data < 10 * n_training:
        training_data_theta_subsets = [
            training_data_theta[np.random.choice(n_samples_data, size=n_training, replace=True)]
            for _ in range(100)
        ]
    else:
        samples = training_data_theta[np.random.choice(n_samples_data, size=n_training * 10, replace=False)]
        training_data_theta_subsets = [samples[i:i + n_training] for i in range(0, len(samples), n_training)]

    # Vectorized mapping functions
    @partial(jnp.vectorize, signature='(i)->(i)', excluded={1})
    def map(x, flow):
        y, _ = flow(x)
        return y

    @partial(jnp.vectorize, signature='(i)->()', excluded={1})
    def distance_to_Gaussian(x, flow):
        y, log_det = flow.inv(x)
        return -jnp.sum(y**2) / 2 + log_det

    def KL_loss_function(flow, x, p):
        distance = distance_to_Gaussian(x, flow)
        return jnp.average(p - distance)

    def Jeffreys_loss_function(flow, x, p):
        distance = distance_to_Gaussian(x, flow)
        KL_div = jnp.average(p - distance)
        rw = jnp.exp(jnp.clip(distance - p, -30.0, 30.0))
        rkl = jnp.sum(rw * (distance - p)) / (jnp.sum(rw) + 1e-8)
        return KL_div + rkl

    # Compute divergences
    KL = KL_loss_function(flow, training_data_theta, training_data_distribution_value)
    Jeffreys = Jeffreys_loss_function(flow, training_data_theta, training_data_distribution_value)
    print(f"Kullback-Leibler divergence: {KL:.5f}")
    print(f"Jeffreys' divergence: {Jeffreys:.5f}")

    new_row = [f"n_training: {n_training}, n_layers: {n_layers}, batch_size_training: {n_batch_size_training}, n_training_steps: {n_training_steps}, KL: {KL:.5f}, Jeffreys: {Jeffreys:.5f}\n"]

    x = jax.random.normal(sample_key, (n_samples, n_dimensions))
    x_nf = map(x, flow)

    def ratio_diff(subsets, stat_fn, x_model, x_nf):
        diffs = []
        for i in range(n_dimensions):
            subset_stats = [stat_fn(s[:, i]) for s in subsets]
            std = np.std(subset_stats)
            model_stat = stat_fn(x_model[:, i])
            nf_stat = stat_fn(x_nf[:, i])
            diff = float(nf_stat - model_stat)
            print(f"{i}th coord: model = {model_stat:.5f} ± {std:.5f}, NF = {nf_stat:.5f}, |Δ/std| = {abs(diff / std):.3f}")
            diffs.append(f"{i}th coord: model = {model_stat:.5f} +- {std:.5f}, NF = {nf_stat:.5f}, |err/std| = {abs(diff / std):.3f}\n")
        return diffs

    if metric_selection[0]:
        m = ratio_diff(training_data_theta_subsets, np.mean, training_data_theta, x_nf)
        new_row.append("Mean:\n")
        for i in range(len(m)):
            new_row.append(m[i])
    else:
        new_row.append("No mean metric computed\n")
    if metric_selection[1]:
        v = ratio_diff(training_data_theta_subsets, np.var, training_data_theta, x_nf)
        new_row.append("Variance:\n")
        for i in range(len(v)):
            new_row.append(v[i])
    else:
        new_row.append("No variance metric computed\n")
    if metric_selection[2]:
        s = ratio_diff(training_data_theta_subsets, skew, training_data_theta, x_nf)
        new_row.append("Skewness:\n")
        for i in range(len(s)):
            new_row.append(s[i])
    else:
        new_row.append("No skew metric computed\n")
    if metric_selection[3]:
        k = ratio_diff(training_data_theta_subsets, kurtosis, training_data_theta, x_nf)
        new_row.append("Kurtosis:\n")
        for i in range(len(k)):
            new_row.append(k[i])
    else:
        new_row.append("No kurtosis metric computed\n")
    
    if metric_selection[4]:
        for i in range(n_dimensions):
            for j in range(i + 1, n_dimensions):
                cov_subsets = [np.cov(s[:, i], s[:, j])[0, 1] for s in training_data_theta_subsets]
                std = np.std(cov_subsets)
                cov_model = np.cov(training_data_theta[:, i], training_data_theta[:, j])[0, 1]
                cov_nf = np.cov(x_nf[:, i], x_nf[:, j])[0, 1]
                diff = float(cov_nf - cov_model)
                print(f"Cov({i},{j}): model = {cov_model:.5f} ± {std:.5f}, NF = {cov_nf:.5f}, Δ/std = {diff / std:.3f}")
                new_row.append(f"Cov({i},{j}): model = {cov_model:.5f} +- {std:.5f}, NF = {cov_nf:.5f}, err/std = {diff / std:.3f}\n")
    else:
        new_row.append("No covariance metric computed\n")

    if metrics_path is None:
        return new_row  # Return metrics list instead of DataFrame
    else:
        file_exists = False
        try:
            with open(metrics_path, 'r') as f:
                file_exists = True
        except FileNotFoundError:
            pass

        with open(metrics_path, 'a') as f:
            if not file_exists:
                # Write a header if the file doesn't exist
                f.write('n_samples, n_layers, batch_size_training, n_training_steps, '
                        'KL, J, mean, var, skew, kurtosis, cov\n')

            # Write each string in new_row as a separate line
            for row in new_row:
                f.write(row + '\n')
