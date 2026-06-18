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
    elif reward_type == 'shaped-sa':
        return DiscriminatorShapedSA


######################################################
######################################################
"""
Discriminators. Available types:
- D(s,a) : state-action
- D(s,s') : state-based
- D(s) + gamma h(s') - h(s) : shaped
- D(s,a) + gamma h(s') - h(s) : shaped-sa

All accept input as s, a, s', absorbing: bool for implementation convenience. All handle absorbing states to avoid termination/survival bias (https://arxiv.org/pdf/1809.02925)
"""
######################################################
######################################################

class DiscriminatorShapedSA(nn.Module):

    hidden_layer_dims: Sequence[int]
    gamma: int
    output_dim: int = 1
    activation: str = "tanh"
    output_activation: str = None    # none means linear activation
    use_running_mean_stand: bool = True
    squeeze_output: bool = True

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)

        g_layers = []
        for dim_layer in self.hidden_layer_dims:
            g_layers.append(nn.Dense(dim_layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)))
        g_layers.append(nn.Dense(self.output_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0)))
        self.g_nets = g_layers

        h_layers = []
        for dim_layer in self.hidden_layer_dims:
            h_layers.append(nn.Dense(dim_layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)))
        h_layers.append(nn.Dense(self.output_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0)))
        self.h_nets = h_layers

    @nn.compact
    def __call__(self, x, a, x_n, absorbing, logp, shaping: float = 1.0):
        """
        D(s) + gamma h(s') - h(s)
        Args:
            x : state
            a: action
            x_n: next state
            abs: bool for whether the next state is absorbing
            logp: log probability of the actions
        """
        x = jnp.concatenate([jnp.atleast_2d(x),
                            jnp.broadcast_to(jnp.reshape(absorbing, (-1,1)), (jnp.atleast_2d(x).shape[0], 1))], axis=1)    
        x_n = jnp.concatenate([jnp.atleast_2d(x_n),
                            jnp.broadcast_to(jnp.reshape(absorbing, (-1,1)), (jnp.atleast_2d(x_n).shape[0], 1))], axis=1)

        xa = jnp.concatenate([jnp.atleast_2d(x), jnp.atleast_2d(a),
                            jnp.broadcast_to(jnp.reshape(absorbing, (-1,1)), (jnp.atleast_2d(x).shape[0], 1))], axis=1)    
        xa_n = jnp.concatenate([jnp.atleast_2d(x_n), jnp.atleast_2d(a),
                            jnp.broadcast_to(jnp.reshape(absorbing, (-1,1)), (jnp.atleast_2d(x_n).shape[0], 1))], axis=1)

        if self.use_running_mean_stand:
            rms = RunningMeanStd()
            rms_xa = RunningMeanStd()
            x = rms(x)
            x_n = rms(x_n)
            xa = rms_xa(xa)
            xa_n = rms_xa(xa_n) 

        # g(x)
        r = xa
        for layer in self.g_nets[:-1]:
            r = layer(r)
            r = self.activation_fn(r)
        r = jnp.squeeze(self.g_nets[-1](r))

        # g(x_n)
        rx_n = xa_n
        for layer in self.g_nets[:-1]:
            rx_n = layer(rx_n)
            rx_n = self.activation_fn(rx_n)
        rx_n = jnp.squeeze(self.g_nets[-1](rx_n))

        # h(x)
        hx = x
        for layer in self.h_nets[:-1]:
            hx = layer(hx)
            hx = self.activation_fn(hx)
        hx = jnp.squeeze(self.h_nets[-1](hx))

        # h(x_n)
        hx_n = x_n
        for layer in self.h_nets[:-1]:
            hx_n = layer(hx_n)
            hx_n = self.activation_fn(hx_n)
        hx_n = jnp.squeeze(self.h_nets[-1](hx_n))

        # Shaped reward: step. 6 in alg 1 in https://arxiv.org/pdf/1710.11248v2
        f = r + shaping * (((1 - absorbing) * self.gamma * hx_n) + (absorbing * (self.gamma/(1 - self.gamma)) * rx_n) - hx)
        reward = f - logp # D = sigmoid(reward) and we compute loss using sigmoid BCE

        return jnp.squeeze(reward) if self.squeeze_output else reward


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

        g_layers = []
        for dim_layer in self.hidden_layer_dims:
            g_layers.append(nn.Dense(dim_layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)))
        g_layers.append(nn.Dense(self.output_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0)))
        self.g_nets = g_layers

        h_layers = []
        for dim_layer in self.hidden_layer_dims:
            h_layers.append(nn.Dense(dim_layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)))
        h_layers.append(nn.Dense(self.output_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0)))
        self.h_nets = h_layers

    @nn.compact
    def __call__(self, x, a, x_n, absorbing, logp, shaping: float = 1.0):
        """
        D(s) + gamma h(s') - h(s)
        Args:
            x : state
            a: action (not used)
            x_n: next state
            abs: bool for whether the next state is absorbing
            logp: log probability of the actions
        """
        x = jnp.concatenate([jnp.atleast_2d(x),
                            jnp.broadcast_to(jnp.reshape(absorbing, (-1,1)), (jnp.atleast_2d(x).shape[0], 1))], axis=1)    
        x_n = jnp.concatenate([jnp.atleast_2d(x_n),
                            jnp.broadcast_to(jnp.reshape(absorbing, (-1,1)), (jnp.atleast_2d(x_n).shape[0], 1))], axis=1)

        if self.use_running_mean_stand:
            rms = RunningMeanStd()
            x   = rms(x)
            x_n = rms(x_n)

        # g(x)
        r = x
        for layer in self.g_nets[:-1]:
            r = layer(r)
            r = self.activation_fn(r)
        r = jnp.squeeze(self.g_nets[-1](r))

        # g(x_n)
        rx_n = x_n
        for layer in self.g_nets[:-1]:
            rx_n = layer(rx_n)
            rx_n = self.activation_fn(rx_n)
        rx_n = jnp.squeeze(self.g_nets[-1](rx_n))

        # h(x)
        hx = x
        for layer in self.h_nets[:-1]:
            hx = layer(hx)
            hx = self.activation_fn(hx)
        hx = jnp.squeeze(self.h_nets[-1](hx))

        # h(x_n)
        hx_n = x_n
        for layer in self.h_nets[:-1]:
            hx_n = layer(hx_n)
            hx_n = self.activation_fn(hx_n)
        hx_n = jnp.squeeze(self.h_nets[-1](hx_n))

        # Shaped reward: step. 6 in alg 1 in https://arxiv.org/pdf/1710.11248v2
        f = r + shaping * (((1 - absorbing) * self.gamma * hx_n) + (absorbing * (self.gamma/(1 - self.gamma)) * rx_n) - hx)
        reward = f - logp # D = sigmoid(reward) and we compute loss using sigmoid BCE

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
    def __call__(self, x, y, x_n, absorbing, logp, shaping=None):
        """
        D(s,a)

        Args:
            x : state
            y: action 
            x_n: next state (not used)
            absorbing: bool for whether the next state is absorbing (not used)
            logp: log probability of the actions (not used)
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
    def __call__(self, x, y, x_n, absorbing, logp, shaping=None):
        """
        D(s, s')

        Args:
            x : state
            y: action (not used)
            x_n: next state
            absorbing: bool for whether the next state is absorbing (not used)
            logp: log probability of the actions (not used)
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
