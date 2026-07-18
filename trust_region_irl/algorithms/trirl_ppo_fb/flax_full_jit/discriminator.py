import numpy as np
from typing import Sequence
import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType
from collections import deque


######################################################
######################################################
"""
All reward functions are linear in latent features. r = theta^T phi(.) where phi(.) returns latent features from either state-action or base features

Discriminators. Available types:
- D(phi(.)) : feature-based
- D(psi(phi(.))) : boltzmann-feature-based
- D(psi(phi(x))) + gamma h(psi(phi(x'))) = h(psi(phi(x))) : shapedboltzmann-feature-based

All accept input as f, s, a, s', absorbing: bool for implementation convenience. All handle absorbing states to avoid termination/survival bias (https://arxiv.org/pdf/1809.02925)
"""
######################################################
######################################################

def get_discriminator(config, env, reward_type='feature-based'):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:       
        if reward_type == 'feature-based':
            return DiscriminatorFeatureBased()
        elif reward_type == 'boltzmann-feature-based':
            hidden_dims = tuple(int(x) for x in str(config.algorithm.boltzmann_hidden_dims).split(",") if x.strip() != "")
            return BoltzmannDiscriminatorFeatureBased(
                hidden_dims=hidden_dims,
                latent_dim=int(config.algorithm.boltzmann_latent_dim),
                energy_hidden_dim=int(config.algorithm.boltzmann_energy_hidden_dim),
            )
        elif reward_type == 'shapedboltzmann-feature-based':
            raise NotImplementedError
        else:
            raise NotImplementedError

class DiscriminatorFeatureBased(nn.Module):
    @nn.compact
    def __call__(self, f, x, a, x_n, absorbing, shaping=None):
        """
        D(s,a) = theta^T phi(.)

        Args:
            f : features
        """
        theta = self.param("theta", constant(0.001), (f.shape[-1],))
        return jnp.dot(f, theta)


class BoltzmannDiscriminatorFeatureBased(nn.Module):
    # architecture is configurable (see default_config.py boltzmann_* fields)
    hidden_dims: Sequence[int] = (4, 8)   # encoder hidden layer widths
    latent_dim: int = 16                  # encoder output dim (= reward theta dim)
    energy_hidden_dim: int = 32           # energy head hidden width

    def setup(self):
        # Layer names kept as feat_dense1..N (last = output) so the DEFAULT (4,8)/16 arch is
        # byte-identical to the old hard-coded module and old checkpoints still load.
        dims = list(self.hidden_dims) + [self.latent_dim]
        layers = []
        for i, d in enumerate(dims):
            is_out = (i == len(dims) - 1)
            layers.append(nn.Dense(d, kernel_init=orthogonal(1.0 if is_out else jnp.sqrt(2)),
                                   bias_init=constant(0.0), name=f"feat_dense{i+1}"))
        self.feat_layers = layers

        self.theta = self.param("theta", constant(0.0), (self.latent_dim,))

        self.energy_dense1 = nn.Dense(self.energy_hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        self.energy_dense2 = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def encode_feature(self, f):
        z = f
        for i, layer in enumerate(self.feat_layers):
            z = layer(z)
            if i < len(self.feat_layers) - 1:   # relu on hidden layers only, not the output
                z = nn.relu(z)
        return z

    def energy_from_z(self, z):
        e = self.energy_dense1(z)
        e = nn.tanh(e)
        e = self.energy_dense2(e)
        return jnp.squeeze(e, axis=-1)

    def energy_only(self, f):
        z = self.encode_feature(f)
        return self.energy_from_z(z)

    def reward_only(self, f):
        z = self.encode_feature(f)
        r = jnp.dot(z, self.theta)
        return r

    def __call__(self, f, x, a, x_n, absorbing, shaping: float = 1.0):
        """
        D(s,a) = theta^T psi(phi(.)) where psi is a feature encoder trained to return Boltzmann features (using denoising score matching)

        Args:
            f : features
        """
        zf = self.encode_feature(f)
        _ = self.energy_from_z(zf)
        r = jnp.dot(zf, self.theta)
        return r
