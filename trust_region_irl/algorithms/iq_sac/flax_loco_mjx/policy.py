from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

import jax.core
jax.interpreters.xla.pytype_aval_mappings = jax.core.pytype_aval_mappings
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from loco_mujoco.algorithms import FullyConnectedNet, RunningMeanStd
from trust_region_irl.algorithms.iq_sac.flax_loco_mjx.tanh_transformed_distribution import TanhTransformedDistribution


class Actor(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    init_std: float = 1.0
    learnable_std: bool = True
    hidden_layer_dims: Sequence[int] = (1024, 512)
    actor_obs_ind: jnp.ndarray = None

    def setup(self):
        self.critic1 = FullyConnectedNet(self.hidden_layer_dims, 1, self.activation, None, False, False)
        self.critic2 = FullyConnectedNet(self.hidden_layer_dims, 1, self.activation, None, False, False)

    @nn.compact
    def __call__(self, x):

        x = RunningMeanStd()(x)

        # build actor
        actor_x = x if self.actor_obs_ind is None else x[..., self.actor_obs_ind]
        actor_mean = FullyConnectedNet(self.hidden_layer_dims, self.action_dim, self.activation,
                                       None, False, False)(actor_x)
        actor_logtstd = self.param("log_std", nn.initializers.constant(jnp.log(self.init_std)),
                                   (self.action_dim,))
        if not self.learnable_std:
            actor_logtstd = jax.lax.stop_gradient(actor_logtstd)

        pi = TanhTransformedDistribution(tfd.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd)))
        return pi


class Critic(nn.Module):
    nr_critics: int
    activation: str = "tanh"
    hidden_layer_dims: Sequence[int] = (1024, 512)

    @nn.compact
    def __call__(self, x, a):

        x = jnp.concatenate([jnp.atleast_2d(x),
                            jnp.atleast_2d(a)], axis=1)
        x = RunningMeanStd()(x)

        vmap_critic = nn.vmap(
            FullyConnectedNet,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.nr_critics,
        )

        q_values = vmap_critic(self.hidden_layer_dims, 1, self.activation, None, False, False)(x)
        return q_values

