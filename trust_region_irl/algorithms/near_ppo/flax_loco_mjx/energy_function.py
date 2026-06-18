import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
from loco_mujoco.algorithms import RunningMeanStd


def get_energyfn(ncsnv1=True):
    if ncsnv1:
        return EnergyFnCond
    else:
        return EnergyFn


class EnergyFnCond(nn.Module):

    encoder_hidden_layer_dims: Sequence[int]
    decoder_hidden_layer_dims: Sequence[int]
    use_running_mean_stand: bool = True
    steps: int = 100


    @nn.compact
    def __call__(self, x, cond):
        """
        D(s,a)

        Args:
            x : sample
        """
        half_dim = self.encoder_hidden_layer_dims[-1]/2
        if self.use_running_mean_stand:
            x = RunningMeanStd()(x)

        # Encoder
        for hidden_dim in self.encoder_hidden_layer_dims:
            x = nn.Dense(hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.LayerNorm()(x)
            x = nn.elu(x)

        # SinusoidalPosEmb
        cond = self.steps * cond
        emb = jnp.log(10000.0) / (half_dim - 1)
        emb = jnp.exp(-emb * jnp.arange(half_dim))
        emb = cond * emb
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        x = x + emb

        # Decoder
        for hidden_dim in self.decoder_hidden_layer_dims:
            x = nn.Dense(hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.LayerNorm()(x)
            x = nn.elu(x)

        # Final projection to scalar
        energyfn = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        energyfn = nn.elu(energyfn)

        return energyfn



class EnergyFn(nn.Module):

    encoder_hidden_layer_dims: Sequence[int]
    decoder_hidden_layer_dims: Sequence[int]
    use_running_mean_stand: bool = True


    @nn.compact
    def __call__(self, x, cond):
        """
        D(s,a)

        Args:
            x : sample
        """
        if self.use_running_mean_stand:
            x = RunningMeanStd()(x)

        x_init = x

        # Encoder
        for hidden_dim in self.encoder_hidden_layer_dims:
            x = nn.Dense(hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.LayerNorm()(x)
            x = nn.gelu(x)

        # Residual 
        x = x + nn.Dense(self.encoder_hidden_layer_dims[-1], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x_init)

        # Decoder
        for hidden_dim in self.decoder_hidden_layer_dims:
            x = nn.Dense(hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.LayerNorm()(x)
            x = nn.gelu(x)

        # Final projection to scalar
        energyfn = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        energyfn = nn.gelu(energyfn)
        energyfn = energyfn / cond

        return energyfn