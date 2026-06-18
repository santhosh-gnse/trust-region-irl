import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence

from loco_mujoco.algorithms import RunningMeanStd
from loco_mujoco.algorithms.common.networks import get_activation_fn


def get_discriminator(reward_type='state-action'):
    if reward_type == 'state-action':
        return Discriminator
    elif reward_type == 'state-based':
        return DiscriminatorStateBased
    elif reward_type == 'shaped':
        return DiscriminatorShaped


def get_reward_approximator(reward_type='state-action'):
    if reward_type == 'state-action':
        return Discriminator
    elif reward_type == 'state-based':
        return DiscriminatorStateBased
    elif reward_type == 'shaped':
        return RewardApproximatorShaped

######################################################
######################################################
"""
Discriminators. Available types:
- D(s,a) : state-action
- D(s,s') : state-based
- D(s) + gamma h(s') - h(s) : shaped

All accept input as s, a, s', absorbing: bool for implementation convenience. All handle absorbing states to avoid termination/survival bias (https://arxiv.org/pdf/1809.02925)
"""
######################################################
######################################################


class DiscriminatorShaped(nn.Module):

    hidden_layer_dims: Sequence[int]
    gamma: int
    output_dim: int = 1
    activation: str = "tanh"
    output_activation: str = None    # none means linear activation
    use_running_mean_stand: bool = True
    squeeze_output: bool = True

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)
        self.output_activation_fn = get_activation_fn(self.output_activation) \
            if self.output_activation is not None else lambda x: x

        g_nets = []
        for i, dim_layer in enumerate(self.hidden_layer_dims):
            g_nets.append(nn.Dense(dim_layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)))
        g_nets.append(nn.Dense(self.output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)))

        h_nets = []
        for i, dim_layer in enumerate(self.hidden_layer_dims):
            h_nets.append(nn.Dense(dim_layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)))
        h_nets.append(nn.Dense(self.output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)))

        self.g_nets, self.h_nets = g_nets, h_nets

    @nn.compact
    def __call__(self, x, a, x_n, absorbing, shaping: float = 1.0):
        """
        D(s) + gamma h(s') - h(s)
        Args:
            x : state
            a: action (not used)
            x_n: next state
            abs: bool for whether the next state is absorbing
        """
        x = jnp.concatenate([jnp.atleast_2d(x),
                            jnp.broadcast_to(jnp.reshape(absorbing, (-1,1)), (jnp.atleast_2d(x).shape[0], 1))], axis=1)    
        x_n = jnp.concatenate([jnp.atleast_2d(x_n),
                            jnp.broadcast_to(jnp.reshape(absorbing, (-1,1)), (jnp.atleast_2d(x_n).shape[0], 1))], axis=1)
        if self.use_running_mean_stand:
            x = RunningMeanStd()(x)
            x_n = RunningMeanStd()(x_n)

        # g(x)
        r = x
        for layer in self.g_nets[:-1]:
            r = layer(r)
            r = self.activation_fn(r)
        r = self.g_nets[-1](r)
        r = jnp.squeeze(self.output_activation_fn(r))

        # g(x_n)
        rx_n = x_n
        for layer in self.g_nets[:-1]:
            rx_n = layer(rx_n)
            rx_n = self.activation_fn(rx_n)
        rx_n = self.g_nets[-1](rx_n)
        rx_n = jnp.squeeze(self.output_activation_fn(rx_n))

        # h(x)
        hx = x
        for layer in self.h_nets[:-1]:
            hx = layer(hx)
            hx = self.activation_fn(hx)
        hx = self.h_nets[-1](hx)
        hx = jnp.squeeze(self.output_activation_fn(hx))

        # h(x_n)
        hx_n = x_n
        for layer in self.h_nets[:-1]:
            hx_n = layer(hx_n)
            hx_n = self.activation_fn(hx_n)
        hx_n = self.h_nets[-1](hx_n)
        hx_n = jnp.squeeze(self.output_activation_fn(hx_n))


        # Shaped reward: step. 6 in alg 1 in https://arxiv.org/pdf/1710.11248v2
        f = r + shaping * ((1 - absorbing) * self.gamma * hx_n + absorbing * ((self.gamma/(1 - self.gamma)) * rx_n) - hx)
        reward = f # D = sigmoid(reward) and we compute loss using sigmoid BCE

        return jnp.squeeze(reward) if self.squeeze_output else reward



class Discriminator(nn.Module):

    hidden_layer_dims: Sequence[int]
    gamma: int # unused
    output_dim: 1
    activation: str = "tanh"
    output_activation: str = None    # none means linear activation
    use_running_mean_stand: bool = True
    squeeze_output: bool = True
    

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)
        self.output_activation_fn = get_activation_fn(self.output_activation) \
            if self.output_activation is not None else lambda x: x

    @nn.compact
    def __call__(self, x, y, x_n, absorbing, shaping=None):
        """
        D(s,a)

        Args:
            x : state
            y: action 
            x_n: next state (not used)
            absorbing: bool for whether the next state is absorbing (not used)
        """
        x = jnp.concatenate([jnp.atleast_2d(x),
                            jnp.broadcast_to(jnp.reshape(absorbing, (-1,1)), (jnp.atleast_2d(x).shape[0], 1)),
                            jnp.atleast_2d(y)], axis=1)

        if self.use_running_mean_stand:
            x = RunningMeanStd()(x)

        # build network
        for i, dim_layer in enumerate(self.hidden_layer_dims):
            x = nn.Dense(dim_layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = self.activation_fn(x)

        # add last layer
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        x = self.output_activation_fn(x)

        return jnp.squeeze(x) if self.squeeze_output else x



class DiscriminatorStateBased(nn.Module):

    hidden_layer_dims: Sequence[int]
    gamma: int # unused
    output_dim: 1
    activation: str = "tanh"
    output_activation: str = None    # none means linear activation
    use_running_mean_stand: bool = True
    squeeze_output: bool = True

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)
        self.output_activation_fn = get_activation_fn(self.output_activation) \
            if self.output_activation is not None else lambda x: x

    @nn.compact
    def __call__(self, x, y, x_n, absorbing, shaping=None):
        """
        D(s, s')

        Args:
            x : state
            y: action (not used)
            x_n: next state
            absorbing: bool for whether the next state is absorbing (not used)
        """
        x = jnp.concatenate([jnp.atleast_2d(x),
                            jnp.broadcast_to(jnp.reshape(absorbing, (-1,1)), (jnp.atleast_2d(x).shape[0], 1)),
                            jnp.atleast_2d(x_n)], axis=1)

        if self.use_running_mean_stand:
            x = RunningMeanStd()(x)

        # build network
        for i, dim_layer in enumerate(self.hidden_layer_dims):
            x = nn.Dense(dim_layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = self.activation_fn(x)

        # add last layer
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        x = self.output_activation_fn(x)

        return jnp.squeeze(x) if self.squeeze_output else x



######################################################
######################################################
"""
Reward Function Approximators. Available types:
- r(s,a) : state-action
- r(s,s') : state-based
- r(s) + gamma h(s') - h(s) : shaped

All accept input as s, a, s', shaping: bool for implementation convenience
"""
######################################################
######################################################

class RewardApproximatorShaped(nn.Module):

    hidden_layer_dims: Sequence[int]
    gamma: int
    output_dim: int = 1
    activation: str = "tanh"
    output_activation: str = None    # none means linear activation
    use_running_mean_stand: bool = True
    squeeze_output: bool = True

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)
        self.output_activation_fn = get_activation_fn(self.output_activation) \
            if self.output_activation is not None else lambda x: x

        g_nets = []
        for i, dim_layer in enumerate(self.hidden_layer_dims):
            g_nets.append(nn.Dense(dim_layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)))
        g_nets.append(nn.Dense(self.output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)))

        h_nets = []
        for i, dim_layer in enumerate(self.hidden_layer_dims):
            h_nets.append(nn.Dense(dim_layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)))
        h_nets.append(nn.Dense(self.output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)))

        self.g_nets, self.h_nets = g_nets, h_nets

    @nn.compact
    def __call__(self, x, a, x_n, absorbing, shaping: float = 1.0):
        """
        D(s) + gamma h(s') - h(s)
        Args:
            x : state
            a: action (not used)
            x_n: next state
            abs: bool for whether the next state is absorbing
        """
        x = jnp.concatenate([jnp.atleast_2d(x),
                            jnp.broadcast_to(jnp.reshape(absorbing, (-1,1)), (jnp.atleast_2d(x).shape[0], 1))], axis=1)    
        x_n = jnp.concatenate([jnp.atleast_2d(x_n),
                            jnp.broadcast_to(jnp.reshape(absorbing, (-1,1)), (jnp.atleast_2d(x_n).shape[0], 1))], axis=1)
        if self.use_running_mean_stand:
            x = RunningMeanStd()(x)
            x_n = RunningMeanStd()(x_n)

        # g(x)
        r = x
        for layer in self.g_nets[:-1]:
            r = layer(r)
            r = self.activation_fn(r)
        r = self.g_nets[-1](r)
        r = jnp.squeeze(self.output_activation_fn(r))

        # h(x)
        hx = x
        for layer in self.h_nets[:-1]:
            hx = layer(hx)
            hx = self.activation_fn(hx)
        hx = self.h_nets[-1](hx)
        hx = jnp.squeeze(self.output_activation_fn(hx))

        # h(x_n)
        hx_n = x_n
        for layer in self.h_nets[:-1]:
            hx_n = layer(hx_n)
            hx_n = self.activation_fn(hx_n)
        hx_n = self.h_nets[-1](hx_n)
        hx_n = jnp.squeeze(self.output_activation_fn(hx_n))

        reward = r + shaping * (self.gamma * hx_n - hx)

        return jnp.squeeze(reward) if self.squeeze_output else reward