import json
from typing import Tuple, Dict

import jax
import equinox as eqx

from .flows import RealNVPScaleShift


def make(hyperparams: Dict, key: jax.random.PRNGKey) -> RealNVPScaleShift:
    """
    Create a RealNVPScaleShift flow based on hyperparameters and random key.

    Args:
        hyperparams: Dictionary containing 'dimension' and 'layers' keys.
        key: PRNGKey for initializing the model.

    Returns:
        An instance of RealNVPScaleShift.
    """
    dim = hyperparams["dimension"]
    depth = hyperparams["layers"]
    return RealNVPScaleShift(key=key, dim=dim, depth=depth)


def save(filename: str, hyperparams: Dict, model: RealNVPScaleShift) -> None:
    """
    Save the flow model and its hyperparameters to a file.

    Args:
        filename: Output file path.
        hyperparams: Dictionary of hyperparameters.
        model: Flow model to save.
    """
    with open(filename, "wb") as f:
        header = json.dumps(hyperparams) + "\n"
        f.write(header.encode("utf-8"))
        eqx.tree_serialise_leaves(f, model)


def load(filename: str, init_key: jax.random.PRNGKey) -> Tuple[RealNVPScaleShift, Dict]:
    """
    Load a flow model and its hyperparameters from a file.

    Args:
        filename: Path to file.
        init_key: Random key used to initialize model structure before deserialization.

    Returns:
        A tuple of (flow model, hyperparameters).
    """
    with open(filename, "rb") as f:
        header = f.readline().decode("utf-8")
        hyperparams = json.loads(header)
        model = make(hyperparams, init_key)
        model = eqx.tree_deserialise_leaves(f, model)
    return model, hyperparams
