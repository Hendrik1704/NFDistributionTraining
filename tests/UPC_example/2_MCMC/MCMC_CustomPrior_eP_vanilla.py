import os
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
import sys

# Get the path three levels up and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.mcmc import Chain # Provide the correct include to the MCMC Chain class here
from src.realnvp.utils import load

# RUN WITH surmise==0.2.1, pocomc==1.2.6 (I use numpy==1.23.5, probably not necessary)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["RDMAV_FORK_SAFE"] = "1"
n_effective=128000
n_active=64000
n_prior=128000
sample="tpcn"
n_max_steps=100
random_state=42

n_total = 100000
n_evidence = 7500
pool = 12

def load_exp_data(exp_path):
    with open(exp_path, "rb") as f:
        exp_data = pickle.load(f)
        exp_data_y = exp_data['000']['obs'][0]
        exp_data_error = exp_data['000']['obs'][1]
    return exp_data_y, exp_data_error

class CustomPrior:
    def __init__(self, path_NF_model, seed=None):
        self.dim = 11
        # Indicate fixed parameters with comments
        self.bounds = np.array([[0.02, 1.2],
                                [1.0, 10.0],
                                [0.05, 3.0],
                                [0.0, 10.0], # Nq = 3
                                [0.0, 1.5],
                                [0.05, 1.5],
                                [0.02, 1.2],
                                [0.0001, 0.28],
                                [0.0, 1.0], # vUV = 0
                                [0.0, 10.0], # omega = 1
                                [0.01, 4.0] # K = 1
                                ])
        
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
        # Delete the fixed parameters from the input data
        # The fixed parameters are Nq, vUV, omega, and K
        x = jnp.delete(x, jnp.array([3, 8, 9, 10]), axis=1)
        log_probab = self.flow.log_prob(x)
        return log_probab

    def rvs(self, size=1):
        # Convert bounds to JAX arrays for vectorized operations
        lower = jnp.array(self.bounds[:, 0])
        upper = jnp.array(self.bounds[:, 1])

        # Output buffer for valid samples
        samples = jnp.zeros((size, self.dim))
        num_accepted = 0
        batch_size = max(2 * size, 1000)
        while num_accepted < size:
            self.sample_key, pkey = jax.random.split(self.sample_key)
            z = jax.random.normal(pkey, (batch_size, self.dimension))
            x_nf, _ = self.flow(z)

            # Insert uniformly sampled fixed parameters
            self.sample_key, *subkeys = jax.random.split(self.sample_key, 5)
            Nq_vals = jax.random.uniform(subkeys[0], (batch_size, 1), minval=lower[3], maxval=upper[3])
            vUV_vals = jax.random.uniform(subkeys[1], (batch_size, 1), minval=lower[8], maxval=upper[8])
            omega_vals = jax.random.uniform(subkeys[2], (batch_size, 1), minval=lower[9], maxval=upper[9])
            K_vals = jax.random.uniform(subkeys[3], (batch_size, 1), minval=lower[10], maxval=upper[10])

            x = jnp.insert(x_nf, 3, Nq_vals.squeeze(), axis=1)
            x = jnp.insert(x, 8, vUV_vals.squeeze(), axis=1)
            x = jnp.insert(x, 9, omega_vals.squeeze(), axis=1)
            x = jnp.insert(x, 10, K_vals.squeeze(), axis=1)

            # Keep only samples within bounds
            in_bounds = jnp.all((x >= lower) & (x <= upper), axis=1)
            x_valid = x[in_bounds]

            n_valid = x_valid.shape[0]
            n_needed = size - num_accepted
            n_copy = min(n_valid, n_needed)

            if n_copy > 0:
                samples = samples.at[num_accepted:num_accepted + n_copy].set(x_valid[:n_copy])
                num_accepted += n_copy
        return samples


def run_MCMC(exp_path, model_par, mcmc_path, emuPathList,
             fix_K_factor=False, fix_omega=False, fix_UVdamping=False, 
             fix_Nq=False):
    mymcmc = Chain(expdata_path=exp_path, 
                    model_parafile=model_par, 
                    mcmc_path=mcmc_path,
                    fix_K_factor=fix_K_factor,
                    fix_omega=fix_omega,
                    fix_UVdamping=fix_UVdamping,
                    fix_Nq=fix_Nq,
                    regulate_likelihood=False,
                    regulate_likelihood_value=10.0,
                    no_emu_cov=True)
    mymcmc.loadEmulator(emuPathList)
    prior_class = CustomPrior(
        path_NF_model="./eP_nf_bs5000_L6_lr1e-03.pkl",
        seed=random_state
    )
    mymcmc.run_pocoMC(n_effective=n_effective, n_active=n_active,
                    n_prior=n_prior, sample=sample,
                    n_max_steps=n_max_steps, random_state=random_state,
                    n_total=n_total, n_evidence=n_evidence, pool=pool,
                    prior=prior_class)



exp_path_full = "./exp_data_JIMWLK_allLOG_ePb.pkl"
model_par = "./IP_DIFF_JIMWLK_prior_range"
emuPathPrefix = "../trained_emulators/"
emuPathList_PCGP_full = [
    emuPathPrefix + "emulator_PCGP_set1.pkl",
    emuPathPrefix + "emulator_PCGP_set2.pkl"
]

run_MCMC(exp_path_full, model_par, "./mcmc_PCGP_eP_prior_vanilla/chain.pkl",
            emuPathList_PCGP_full,
            fix_K_factor=True,
            fix_omega=True,
            fix_UVdamping=True,
            fix_Nq=True)