"""Frozen state encoders for linear TRIRL reward features (see scripts/ssl/pretrain_ssl.py)."""
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn


class Encoder(nn.Module):
    """Shared encoder: 256-LN-GELU-256-LN-GELU-d."""
    d: int = 16

    @nn.compact
    def __call__(self, x):
        x = nn.gelu(nn.LayerNorm()(nn.Dense(256)(x)))
        x = nn.gelu(nn.LayerNorm()(nn.Dense(256)(x)))
        return nn.Dense(self.d)(x)


class FrozenFeatures:
    """phi(s) = clip(standardise(f(standardise(s)))), with every statistic fixed at pretraining time.
    Parameters are constants, so nothing in TRIRL can update the encoder."""

    def __init__(self, path):
        with open(path, "rb") as f:
            e = pickle.load(f)
        self.method, self.d = e["method"], int(e["d"])
        self._enc = Encoder(self.d)
        self._params = jax.tree_util.tree_map(jnp.asarray, e["feature_params"][e["feature_key"]])
        self._obs_mean, self._obs_std = jnp.asarray(e["obs_mean"]), jnp.asarray(e["obs_std"])
        self._f_mean, self._f_std = jnp.asarray(e["feat_mean"]), jnp.asarray(e["feat_std"])
        self._lo, self._hi = jnp.asarray(e["feat_lo"]), jnp.asarray(e["feat_hi"])
        self.obs_dim = int(np.asarray(e["obs_mean"]).shape[0])

    def __call__(self, raw_obs):
        x = (raw_obs[..., :self.obs_dim] - self._obs_mean) / self._obs_std
        z = self._enc.apply(jax.lax.stop_gradient(self._params), x)
        return jnp.clip((z - self._f_mean) / self._f_std, self._lo, self._hi)
