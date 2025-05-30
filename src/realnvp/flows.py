from typing import Optional, Tuple
import jax
import jax.numpy as jnp
import equinox as eqx

from .layers import MLP, Linear, ScaleShift

class AffineCoupling(eqx.Module):
    """Affine Coupling layer used in RealNVP."""
    mask: jnp.ndarray
    scale: MLP
    trans: Linear

    def __init__(self, key: jax.random.PRNGKey, dim: int, mask: Optional[jnp.ndarray] = None):
        mask_key, scale_key = jax.random.split(key)
        if mask is None:
            mask = jax.random.randint(mask_key, shape=(dim,), minval=0, maxval=2)
        self.mask = mask
        self.scale = MLP(dim, dim, depth=1, width=1, activation=jnp.tanh, final_activation=lambda x: x, key=scale_key)
        self.trans = Linear(dim, dim, jnp.zeros((dim, dim)), jnp.zeros(dim))

    def map(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the affine coupling transformation."""
        return self(x)[0]

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply the affine coupling transformation and return the output and log determinant."""
        y = self.mask * x
        s = self.scale(y)
        t = self.trans(y)
        z = y + (1.0 - self.mask) * (jnp.exp(s) * x + t)
        log_det = jnp.sum((1.0 - self.mask) * s)
        return z, log_det

    def inv(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply the inverse of the affine coupling transformation and return the output and log determinant."""
        y = self.mask * x
        s = self.scale(y)
        t = self.trans(y)
        z = y + (1.0 - self.mask) * (x - t) * jnp.exp(-s)
        log_det = -jnp.sum((1.0 - self.mask) * s)
        return z, log_det


class CheckeredAffines(eqx.Module):
    """Two-layer checkerboard-style affine couplings."""
    even: AffineCoupling
    odd: AffineCoupling

    def __init__(self, key: jax.random.PRNGKey, dim: int):
        key_even, key_odd = jax.random.split(key)
        mask_even = jnp.arange(dim) % 2
        mask_odd = 1 - mask_even
        self.even = AffineCoupling(key_even, dim, mask=mask_even)
        self.odd = AffineCoupling(key_odd, dim, mask=mask_odd)

    def map(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the transformation to the input."""
        return self(x)[0]

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply the transformation and return the output and log determinant."""
        x, ld_even = self.even(x)
        x, ld_odd = self.odd(x)
        return x, ld_even + ld_odd

    def inv(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply the inverse transformation and return the output and log determinant."""
        x, ld_odd = self.odd.inv(x)
        x, ld_even = self.even.inv(x)
        return x, ld_even + ld_odd


class RealNVP(eqx.Module):
    """Stacked RealNVP composed of multiple CheckeredAffines."""
    dim: int
    depth: int
    layers: list

    def __init__(self, key: jax.random.PRNGKey, dim: int, depth: int):
        self.dim = dim
        self.depth = depth
        keys = jax.random.split(key, depth)
        self.layers = [CheckeredAffines(k, dim) for k in keys]

    def map(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the transformation to the input."""
        return self(x)[0]

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply the transformation and return the output and log determinant."""
        log_det = 0.0
        for layer in self.layers:
            x, ld = layer(x)
            log_det += ld
        return x, log_det

    def inv(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply the inverse transformation and return the output and log determinant."""
        log_det = 0.0
        for i in range(self.depth):
            x, ld = self.layers[-i-1].inv(x)
            log_det += ld
        return x, log_det


class RealNVPScaleShift(eqx.Module):
    """RealNVP with optional ScaleShift transformation for preprocessing."""
    ss: ScaleShift
    rnvp: RealNVP

    def __init__(self, key: jax.random.PRNGKey, dim: int, depth: int,
                 scale: Optional[jnp.ndarray] = None, shift: Optional[jnp.ndarray] = None):
        if scale is None:
            scale = jnp.ones((dim,))
        if shift is None:
            shift = jnp.zeros((dim,))
        self.ss = ScaleShift(dim, scale, shift)
        self.rnvp = RealNVP(key, dim, depth)

    def _forward(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Single-sample forward pass."""
        x, ld_rnvp = self.rnvp(x)
        x, ld_ss = self.ss(x)
        return x, ld_rnvp + ld_ss

    def _inverse(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Single-sample inverse pass."""
        x, ld_ss = self.ss.inv(x)
        x, ld_rnvp = self.rnvp.inv(x)
        return x, ld_rnvp + ld_ss

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Vectorized forward pass for input shape (dim,) or (N, dim)."""
        if x.ndim == 1:
            return self._forward(x)
        elif x.ndim == 2:
            return jax.vmap(self._forward)(x)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

    def inv(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Vectorized inverse pass for input shape (dim,) or (N, dim)."""
        if x.ndim == 1:
            return self._inverse(x)
        elif x.ndim == 2:
            return jax.vmap(self._inverse)(x)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute log probability density under the NF model. Works with (dim,) or (N, dim)."""
        z, log_det = self.inv(x)
        if z.ndim == 1:
            log_pz = -0.5 * jnp.sum(z**2 + jnp.log(2 * jnp.pi))
        else:
            log_pz = -0.5 * jnp.sum(z**2 + jnp.log(2 * jnp.pi), axis=1)
        return log_pz + log_det
