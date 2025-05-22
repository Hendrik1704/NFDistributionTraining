import jax.numpy as jnp
import equinox as eqx
import jax
from typing import Callable

class ScaleShift(eqx.Module):
    """Affine transformation to scale and shift data."""
    dim: int
    scale: jnp.ndarray
    shift: jnp.ndarray

    def __init__(self, dim: int, scale: jnp.ndarray, shift: jnp.ndarray):
        self.dim = dim
        self.scale = jnp.asarray(scale)
        self.shift = jnp.asarray(shift)

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Scale the O(1) distribution back to the original distribution."""
        y = x * self.scale + self.shift
        logdet = jnp.sum(jnp.log(self.scale))
        return y, logdet

    def inv(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Normalize the distribution to have 0 mean and 1 standard deviation."""
        y = (x - self.shift) / self.scale
        logdet = -jnp.sum(jnp.log(self.scale))
        return y, logdet


class Linear(eqx.Module):
    """Linear transformation with fixed weight and bias."""
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_features: int, out_features: int, weight: jnp.ndarray, bias: jnp.ndarray):
        if weight.shape != (out_features, in_features) or bias.shape != (out_features,):
            raise ValueError("Weight or bias shape does not match expected dimensions.")
        self.weight = weight
        self.bias = bias

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.weight @ x + self.bias


class MLP(eqx.Module):
    """Simple MLP using Linear layers."""
    in_dim: int
    out_dim: int
    depth: int
    width: int
    layers: list
    activation: Callable
    final_activation: Callable

    def __init__(self, in_dim: int, out_dim: int, depth: int, width: int, activation: Callable, final_activation: Callable, key):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.depth = depth
        self.width = width
        self.activation = activation
        self.final_activation = final_activation
        self.layers = []
        W = in_dim * width
        keys = jax.random.split(key, depth + 1)
        if depth == 0:
            self.layers.append(Linear(in_dim, out_dim, jnp.zeros((out_dim, in_dim)), jnp.zeros(out_dim)))
        else:
            self.layers.append(Linear(in_dim, W, jax.random.normal(keys[0], (W, in_dim)), jnp.zeros(W)))
            for i in range(1, depth):
                self.layers.append(Linear(W, W, jax.random.normal(keys[i], (W, W)), jnp.zeros(W)))
            self.layers.append(Linear(W, out_dim, jnp.zeros((out_dim, W)), jnp.zeros(out_dim)))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers[:-1]:
            x = layer(x) / jnp.sqrt(self.width)
            x = self.activation(layer(x))
        x = self.final_activation(x)
        x = self.layers[-1](x) / jnp.sqrt(self.out_dim)
        return x