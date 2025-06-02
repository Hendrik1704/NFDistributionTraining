#!/usr/bin/env python

from functools import partial
import os.path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pickle

from .realnvp.flows import RealNVPScaleShift
from .realnvp.utils import save, load
from .metrics_nf import metrics

def train(training_data_path, dimension, normalizing_flow_model_path, 
          test_data_path, batch_size_training=1000, layers=6, 
          learning_rate=1e-3, number_training_steps=10000, seed=None, 
          use_KL_divergence_loss=False, initial_normalizing_flow_model_path=None, 
          loss_threshold_early_stopping=0, optimizer=optax.adam,
          print_loss_training=False, metrics_mid_training=False,
          metrics_mid_training_frequency=1, metrics_path='metrics.dat',
          unsupervised_training=False):
    """
    Train a normalizing flow model using the provided training data and 
    hyperparameters.

    Parameters
    ----------
    training_data_path : str
        Path to the training data file.
    dimension : int
        Dimension of the distribution to train on.
    normalizing_flow_model_path : str
        Path to save the trained normalizing flow model.
    test_data_path : str
        Path to the test data file.
    batch_size_training : int, optional
        Batch size for training step (default is 1000).
    layers : int, optional
        Number of layers in the normalizing flow model (default is 6).
    learning_rate : float, optional
        Learning rate for the optimizer (default is 1e-3).
    number_training_steps : int, optional
        Maximum number of training steps (default is 10000).
    seed : int, optional
        Random seed for reproducibility (default is None).
    use_KL_divergence_loss : bool, optional
        If True, use KL divergence loss instead of negative log likelihood (default is False).
    initial_normalizing_flow_model_path : str, optional
        Path to the initial normalizing flow model for transfer learning (default is None).
    loss_threshold_early_stopping : float, optional
        Threshold for early stopping based on loss (default is 0).
    optimizer : optax.GradientTransformation, optional
        Optimizer for training (default is optax.adam).
    print_loss_training : bool, optional
        If True, print the loss during training (default is False).
    metrics_mid_training : bool, optional
        If True, save metrics during training (default is False).
    metrics_mid_training_frequency : int, optional
        Frequency of saving metrics during training (default is 1).
    metrics_path : str, optional
        Path to save the metrics file (default is 'metrics.dat').
    unsupervised_training : bool, optional
        If True, use unsupervised training without distribution values / 
        log-likelihood in last column of training data (default is False).
    """
    # Specify to use CPU, not GPU for the training
    jax.config.update("jax_platform_name", "cpu")

    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_debug_infs", True)

    # Set the random seed for reproducibility
    if seed is None:
        seed = time.time_ns()
    sample_key, initial_key = jax.random.split(jax.random.PRNGKey(seed), 2)

    if unsupervised_training:
        metrics_mid_training = False
        print("Unsupervised training: metrics can not be computed, "
              "set metrics_mid_training to False")

    with open(training_data_path, 'rb') as pf:
        training_data_raw = pickle.load(pf)

    with open(test_data_path, 'rb') as pf:
        test_data_raw = pickle.load(pf)

    if not unsupervised_training:
        training_data_theta = jnp.array(training_data_raw[:, :-1])
        training_data_distribution_value = jnp.array(training_data_raw[:, -1])
        test_data_theta = jnp.array(test_data_raw[:, :-1])
        test_data_distribution_value = jnp.array(test_data_raw[:, -1])
    else:
        training_data_theta = jnp.array(training_data_raw)
        test_data_theta = jnp.array(test_data_raw)
    n_samples = len(training_data_theta)

    # Check the input data for NaN or Inf values
    assert not jnp.isnan(training_data_theta).any(), "NaN in training data"
    assert not jnp.isinf(training_data_theta).any(), "Inf in training data"
    assert not jnp.isnan(test_data_theta).any(), "NaN in test data"
    assert not jnp.isinf(test_data_theta).any(), "Inf in test data"

    # Preparation for standardization of the training data
    dimension_training_data = training_data_theta.shape[1]
    mean_training_data = jnp.mean(training_data_theta, axis=0)
    std_training_data = jnp.std(training_data_theta, axis=0)

    if initial_normalizing_flow_model_path is None:
        normalizing_flow_model = RealNVPScaleShift(initial_key, 
                                                   dimension, 
                                                   layers,
                                                   std_training_data,
                                                   mean_training_data)
        if print_loss_training:
            print("Training a new normalizing flow model")
    elif os.path.isfile(initial_normalizing_flow_model_path):
        normalizing_flow_model, hyperparameters = load(
            initial_normalizing_flow_model_path, initial_key)
        loaded_dimension = hyperparameters["dim"]
        print("Loaded normalizing flow model with dimension:", loaded_dimension)
        if loaded_dimension != dimension or loaded_dimension != dimension_training_data:
            raise ValueError(
                f"Loaded model dimension {loaded_dimension} does not match "
                f"specified dimension {dimension} or training data dimension {dimension_training_data}")
        else:
            dimension = loaded_dimension
            layers = hyperparameters["layers"]
    else:
        raise FileNotFoundError(
            f"File {initial_normalizing_flow_model_path} does not exist")
    
    # Throw error if the dimension of the training data does not match the dimension of the normalizing flow model
    if dimension_training_data != dimension:
        raise ValueError(
            f"Training data dimension {dimension_training_data} does not match "
            f"specified dimension {dimension}")
    # Throw error if the number of samples in the training data is less than the batch size
    if n_samples < batch_size_training:
        raise ValueError(
            f"Number of samples in the training data {n_samples} is less than "
            f"the batch size {batch_size_training}")
    
    if print_loss_training:
        print("###################################")
        print("Training normalizing flow model")
        print("Dimension:", dimension)
        print(f"Real NVP with {layers} layers followed by a scale-shift transformation")
        print(f"{n_samples} samples in the training data, use batch size {batch_size_training}")
        print(f"Adam optimizer with learning rate {learning_rate:.5f}")
        print(f"Normalizing flow training steps: {number_training_steps}, or until loss < {loss_threshold_early_stopping:.5f}")

    # Effective density function after the normalizing flow
    @partial(jnp.vectorize, signature='(i)->()', excluded={1})
    def distance_to_Gaussian(x, flow):
        """
        Compute the distance to Gaussian distribution using the normalizing flow.

        Parameters
        ----------
        x : jnp.ndarray
            Input data.
        flow : RealNVPScaleShift
            Normalizing flow model.

        Returns
        -------
        jnp.ndarray
            Distance to Gaussian distribution.

        Notes
        -----
        The factor 2 * np.pi is dropped in the loss function.
        """
        y, log_det = flow.inv(x)
        return -jnp.sum(y**2) / 2 + log_det
    
    def KL_loss_function(flow, x, p):
        """
        Compute the KL divergence loss function.

        Parameters
        ----------
        flow : RealNVPScaleShift
            Normalizing flow model.
        x : jnp.ndarray
            Input data.
        p : jnp.ndarray
            Target distribution values.
        
        Returns
        -------
        jnp.ndarray
            KL divergence loss.
        """
        distance = distance_to_Gaussian(x, flow)
        loss = p - distance
        return jnp.average(loss)
    
    def Jeffreys_loss_function(flow, x, p):
        """
        Compute the Jeffreys divergence loss function.

        Parameters
        ----------
        flow : RealNVPScaleShift
            Normalizing flow model.
        x : jnp.ndarray
            Input data.
        p : jnp.ndarray
            Target distribution values.
        
        Returns
        -------
        jnp.ndarray
            Jeffreys divergence loss.
        """
        distance = distance_to_Gaussian(x, flow)
        KL_divergence = jnp.average(p - distance)
        # Prevent over- and underflow in the exponent
        max_exponent, min_exponent = 30.0, -30.0
        rw = jnp.exp(jnp.clip(distance - p, min_exponent, max_exponent))
        numerator = jnp.sum(rw * (distance - p))
        denominator = jnp.sum(rw)
        rkl = numerator / (denominator + 1e-8)
        return KL_divergence + rkl

    def NLL_loss_function(flow, x):
        """
        Compute the negative log-likelihood loss function.

        Parameters
        ----------
        flow : RealNVPScaleShift
            Normalizing flow model.
        x : jnp.ndarray
            Input data.

        Returns
        -------
        jnp.ndarray
            Negative log-likelihood loss.
        """
        y, log_det = flow.inv(x)
        log_prob = -0.5 * jnp.sum(y**2, axis=-1) - 0.5 * x.shape[1] * jnp.log(2 * jnp.pi)
        return -jnp.mean(log_prob + log_det)

    # Compute the loss function and the derivative with respect to the NN parameters
    if unsupervised_training:
        loss_gradient = eqx.filter_value_and_grad(NLL_loss_function)
    elif use_KL_divergence_loss:
        loss_gradient = eqx.filter_value_and_grad(KL_loss_function)
    else:
        loss_gradient = eqx.filter_value_and_grad(Jeffreys_loss_function)

    # Initialization of the optimizer
    opt = optimizer(learning_rate)
    opt_state = opt.init(eqx.filter(normalizing_flow_model, eqx.is_array))

    @eqx.filter_jit
    def step(flow, opt_state, *, key):
        k, key = jax.random.split(key)
        rand_choice = jax.random.choice(k, n_samples, [batch_size_training,], 
                                        replace=False)
        if unsupervised_training:
            loss, gradient = loss_gradient(flow, training_data_theta[rand_choice])
        else:
            loss, gradient = loss_gradient(flow, training_data_theta[rand_choice],
                                   training_data_distribution_value[rand_choice])
        updates, opt_state = opt.update(gradient, opt_state)
        flow = eqx.apply_updates(flow, updates)
        return flow, loss, opt_state, key
    
    # Define hyperparameters in a dictionary
    hyperparameters = {'training_data_path': training_data_path, 
                   'dimension': dimension, 
                   'normalizing_flow_model_path': normalizing_flow_model_path, 
                   'n_samples': n_samples, 
                   'layers': layers, 
                   'batch_size_training': batch_size_training, 
                   'number_training_steps': number_training_steps, 
                   'learning_rate': learning_rate, 
                   'seed': seed, 
                   'initial_normalizing_flow_model_path': initial_normalizing_flow_model_path, 
                   'loss_threshold_early_stopping': loss_threshold_early_stopping}
    loss, test_loss, training_step = 1e2, 1e2, 0
    if metrics_mid_training:
        mid_training_metrics_df = metrics((normalizing_flow_model, hyperparameters), test_data_path, metrics_path=None, seed=None, n_samples=None)
    try:
        if print_loss_training:
            print_function = lambda loss, training_step: print(
                f"Training step {training_step}: loss = {loss:.10f} | Test loss = {test_loss:.10f}")
        else:
            print_function = lambda loss, training_step: None
        while (loss > loss_threshold_early_stopping or test_loss > loss_threshold_early_stopping) and training_step < number_training_steps:
            normalizing_flow_model, loss, opt_state, sample_key = step(
                normalizing_flow_model, opt_state, key=sample_key)
            if unsupervised_training:
                test_loss = NLL_loss_function(normalizing_flow_model, test_data_theta)
            else:
                test_loss = Jeffreys_loss_function(normalizing_flow_model, test_data_theta, test_data_distribution_value)
            training_step += 1
            if metrics_mid_training and training_step % metrics_mid_training_frequency == 0:
                mid_training_metrics_df.append(
                    metrics((normalizing_flow_model, hyperparameters), test_data_path, metrics_path=None, seed=None, n_samples=None))
            print_function(loss, training_step)
    except KeyboardInterrupt:
        print("Training interrupted by user")
        pass

    # Update and save the normalizing flow model and hyperparameters
    number_training_steps = training_step
    hyperparameters['number_training_steps'] = number_training_steps
    save(normalizing_flow_model_path, hyperparameters, normalizing_flow_model)

    if metrics_mid_training:
        with open(metrics_path, 'w') as f:
            for entry in mid_training_metrics_df:
                if isinstance(entry, list):
                    for line in entry:
                        f.write(line)
                else:
                    f.write(str(entry))