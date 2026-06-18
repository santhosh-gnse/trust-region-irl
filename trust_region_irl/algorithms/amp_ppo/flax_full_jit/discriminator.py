import numpy as np
import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType
from collections import deque


######################################################
######################################################
"""
Discriminators. Available types:
- D(s,a) : state-action
- D(s,s') : state-based
- D_0(s) + D_1(a) : uncorrelated
- D(s) + gamma h(s') - h(s) : shaped
- D(s,a) + gamma h(s') - h(s) : shaped-sa

All accept input as s, a, s', absorbing: bool for implementation convenience. All handle absorbing states to avoid termination/survival bias (https://arxiv.org/pdf/1809.02925)
"""
######################################################
######################################################

def get_discriminator(config, env, reward_type='state-action'):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        if reward_type == 'state-action':
            return Discriminator(config.algorithm.nr_hidden_units_disc)
        elif reward_type == 'state-based':
            return DiscriminatorStateBased(config.algorithm.nr_hidden_units_disc)
        elif reward_type == 'shaped':
            return DiscriminatorShaped(config.algorithm.nr_hidden_units_disc, config.algorithm.gamma)
        elif reward_type == 'shaped-sa':
            return DiscriminatorShapedSA(config.algorithm.nr_hidden_units_disc, config.algorithm.gamma)
        elif reward_type == 'uncorrelated':
            return DiscriminatorUncorrelated(config.algorithm.nr_hidden_units_disc)


class Discriminator(nn.Module):
    nr_hidden_units_disc: int

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
        x = jnp.concatenate([x.flatten(), absorbing.flatten(), y.flatten()])
        discriminator = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        discriminator = nn.tanh(discriminator)
        discriminator = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(discriminator)
        discriminator = nn.tanh(discriminator)
        discriminator = nn.Dense(1, kernel_init=orthogonal(0.1), bias_init=constant(0.0))(discriminator)
        return discriminator

    
class DiscriminatorStateBased(nn.Module):
    nr_hidden_units_disc: int

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
        x = jnp.concatenate([x.flatten(), x_n.flatten()])
        discriminator = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        discriminator = nn.relu(discriminator)
        discriminator = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(discriminator)
        discriminator = nn.relu(discriminator)
        discriminator = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(discriminator)
        return discriminator


class DiscriminatorShaped(nn.Module):
    nr_hidden_units_disc: int
    gamma: int

    def setup(self):
        self.gnet_dense1 = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.gnet_dense2 = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.gnet_dense3 = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))

        self.hnet_dense1 = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.hnet_dense2 = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.hnet_dense3 = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))

    def __call__(self, x, a, x_n, absorbing, shaping: float = 1.0):
        """
        D(s) + gamma h(s') - h(s)
        Args:
            x : state
            a: action (not used)
            x_n: next state
            abs: bool for whether the next state is absorbing
        """
        # g(x)
        r = self.gnet_dense1(x)
        r = nn.relu(r)
        r = self.gnet_dense2(r)
        r = nn.relu(r)
        r = self.gnet_dense3(r)

        # g(x_n)
        rx_n = self.gnet_dense1(x_n)
        rx_n = nn.relu(rx_n)
        rx_n = self.gnet_dense2(rx_n)
        rx_n = nn.relu(rx_n)
        rx_n = self.gnet_dense3(rx_n)

        # h(x)
        hx = self.hnet_dense1(x)
        hx = nn.relu(hx)
        hx = self.hnet_dense2(hx)
        hx = nn.relu(hx)
        hx = self.hnet_dense3(hx)

        # h(x_n)
        hx_n = self.hnet_dense1(x_n)
        hx_n = nn.relu(hx_n)
        hx_n = self.hnet_dense2(hx_n)
        hx_n = nn.relu(hx_n)
        hx_n = self.hnet_dense3(hx_n)

        # Shaped reward: step. 6 in alg 1 in https://arxiv.org/pdf/1710.11248v2
        f = r + shaping * ((1 - absorbing) * self.gamma * hx_n + absorbing * ((self.gamma/(1 - self.gamma)) * rx_n) - hx)
        reward = f - shaping * logp # D = sigmoid(reward) and we compute loss using sigmoid BCE

        return reward


class DiscriminatorShapedSA(nn.Module):
    nr_hidden_units_disc: int
    gamma: int

    def setup(self):
        self.gnet_dense1 = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.gnet_dense2 = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.gnet_dense3 = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))

        self.hnet_dense1 = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.hnet_dense2 = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.hnet_dense3 = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))

    def __call__(self, x, a, x_n, absorbing, shaping: float = 1.0):
        """
        D(s,a) + gamma h(s') - h(s)

        Args:
            x : state
            a: action
            x_n: next state
            abs: bool for whether the next state is absorbing
        """
        xa = jnp.concatenate([x.flatten(), a.flatten()])
        xa_n = jnp.concatenate([x.flatten(), 0.0*a.flatten()]) # only used if absorbing is true

        # g(xa)
        r = self.gnet_dense1(xa)
        r = nn.relu(r)
        r = self.gnet_dense2(r)
        r = nn.relu(r)
        r = self.gnet_dense3(r)

        # g(xa_n)
        rxa_n = self.gnet_dense1(xa_n)
        rxa_n = nn.relu(rxa_n)
        rxa_n = self.gnet_dense2(rxa_n)
        rxa_n = nn.relu(rxa_n)
        rxa_n = self.gnet_dense3(rxa_n)

        # h(x)
        hx = self.hnet_dense1(x)
        hx = nn.relu(hx)
        hx = self.hnet_dense2(hx)
        hx = nn.relu(hx)
        hx = self.hnet_dense3(hx)

        # h(x_n)
        hx_n = self.hnet_dense1(x_n)
        hx_n = nn.relu(hx_n)
        hx_n = self.hnet_dense2(hx_n)
        hx_n = nn.relu(hx_n)
        hx_n = self.hnet_dense3(hx_n)

        # AIRL reward: step. 6 in alg 1 in https://arxiv.org/pdf/1710.11248v2
        reward = r + shaping * ((1 - absorbing) * self.gamma * hx_n + absorbing * ((self.gamma/(1 - self.gamma)) * rxa_n) - hx)
        # reward = r + (1 - absorbing) * self.gamma * hx_n - hx

        return reward



class DiscriminatorUncorrelated(nn.Module):
    nr_hidden_units_disc: int

    @nn.compact
    def __call__(self, x, y, x_n, absorbing, shaping=None):

        """
        a * D_0(s,a) + b * D_1(a) + bias

        Args:
            x : state
            y: action
            x_n: next state (not used)
            abs: bool for whether the next state is absorbing (not used)
        """
        x = x.flatten()
        y = y.flatten()
        
        # state reward
        discriminator_x = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        discriminator_x = nn.relu(discriminator_x)
        discriminator_x = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(discriminator_x)
        discriminator_x = nn.LayerNorm()(discriminator_x)
        discriminator_x = nn.relu(discriminator_x)
        discriminator_x = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(discriminator_x)

        # action reward
        discriminator_y = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(y)
        discriminator_y = nn.relu(discriminator_y)
        discriminator_y = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(discriminator_y)
        discriminator_y = nn.LayerNorm()(discriminator_y)
        discriminator_y = nn.relu(discriminator_y)
        discriminator_y = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(discriminator_y)

        # combined
        alpha_x = self.param('alpha_x', nn.initializers.ones, ())
        alpha_y = self.param('alpha_y', nn.initializers.ones, ())
        bias    = self.param('bias',    nn.initializers.zeros, ())
        discriminator = alpha_x * discriminator_x + alpha_y * discriminator_y + bias

        return discriminator


######################################################
######################################################
"""
Reward Function Approximators. Available types:
- r(s,a) : state-action
- r(s,s') : state-based
- r(s) + gamma h(s') - h(s) : shaped
- r(s,a) + gamma h(s') - h(s) : shaped-sa

All accept input as s, a, s', shaping: bool for implementation convenience
"""
######################################################
######################################################

def get_reward_approximator(config, env, reward_approximator_type='state-action'):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        if reward_approximator_type == 'shaped':
            return RewardApproximatorShaped(config.algorithm.nr_hidden_units_disc, config.algorithm.gamma)
        elif reward_approximator_type == 'shaped-sa':
            return RewardApproximatorShapedSA(config.algorithm.nr_hidden_units_disc, config.algorithm.gamma)
        elif reward_approximator_type == 'state-based':
            return RewardApproximatorStateBased(config.algorithm.nr_hidden_units_disc)
        elif reward_approximator_type == 'state-action':
            return RewardApproximator(config.algorithm.nr_hidden_units_disc)
        elif reward_approximator_type == 'state-only':
            return RewardApproximatorStateOnly(config.algorithm.nr_hidden_units_disc)


class RewardApproximatorStateOnly(nn.Module):
    nr_hidden_units_disc: int

    @nn.compact
    def __call__(self, x, y, xn, shaping=None):
        """
        r(s,s')
        Args:
            x : state
            y: action (not used)
            x_n: next state (not used)
        """
        x = x.flatten()
        reward = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        reward = nn.relu(reward)
        reward = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(reward)
        reward = nn.relu(reward)
        reward = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(reward)
        return reward


class RewardApproximator(nn.Module):
    nr_hidden_units_disc: int

    @nn.compact
    def __call__(self, x, y, xn, shaping=None):
        """
        r(s,a)
        Args:
            x : state
            y: action
            x_n: next state (not used)
        """
        x = jnp.concatenate([x.flatten(), y.flatten()])
        reward = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        reward = nn.relu(reward)
        reward = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(reward)
        reward = nn.relu(reward)
        reward = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(reward)
        return reward


class RewardApproximatorStateBased(nn.Module):
    nr_hidden_units_disc: int

    @nn.compact
    def __call__(self, x, y, xn, shaping=None):
        """
        r(s,s')
        Args:
            x : state
            y: action (not used)
            x_n: next state
        """
        x = jnp.concatenate([x.flatten(), xn.flatten()])
        reward = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        reward = nn.relu(reward)
        reward = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(reward)
        reward = nn.relu(reward)
        reward = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(reward)
        return reward


class RewardApproximatorShaped(nn.Module):
    nr_hidden_units_disc: int
    gamma: int

    def setup(self):
        self.gnet_dense1 = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.gnet_dense2 = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.gnet_dense3 = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))

        self.hnet_dense1 = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.hnet_dense2 = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.hnet_dense3 = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))

    def __call__(self, x, a, x_n, shaping: float = 1.0):
        """
        r(s) + gamma h(s') - h(s)
        Args:
            x : state
            a: action (not used)
            x_n: next state
        """
        # g(x)
        r = self.gnet_dense1(x)
        r = nn.relu(r)
        r = self.gnet_dense2(r)
        r = nn.relu(r)
        r = self.gnet_dense3(r)

        # h(x)
        hx = self.hnet_dense1(x)
        hx = nn.relu(hx)
        hx = self.hnet_dense2(hx)
        hx = nn.relu(hx)
        hx = self.hnet_dense3(hx)

        # h(x_n)
        hx_n = self.hnet_dense1(x_n)
        hx_n = nn.relu(hx_n)
        hx_n = self.hnet_dense2(hx_n)
        hx_n = nn.relu(hx_n)
        hx_n = self.hnet_dense3(hx_n)

        reward = r + shaping * (self.gamma * hx_n - hx)

        return reward


class RewardApproximatorShapedSA(nn.Module):
    nr_hidden_units_disc: int
    gamma: int

    def setup(self):
        self.gnet_dense1 = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.gnet_dense2 = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.gnet_dense3 = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))

        self.hnet_dense1 = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.hnet_dense2 = nn.Dense(self.nr_hidden_units_disc, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.hnet_dense3 = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))

    def __call__(self, x, a, x_n, shaping: float = 1.0):
        """
        r(s,a) + gamma h(s') - h(s)
        Args:
            x : state
            a: action
            x_n: next state
        """
        xa = jnp.concatenate([x.flatten(), a.flatten()])

        # g(xa)
        r = self.gnet_dense1(xa)
        r = nn.relu(r)
        r = self.gnet_dense2(r)
        r = nn.relu(r)
        r = self.gnet_dense3(r)

        # h(x)
        hx = self.hnet_dense1(x)
        hx = nn.relu(hx)
        hx = self.hnet_dense2(hx)
        hx = nn.relu(hx)
        hx = self.hnet_dense3(hx)

        # h(x_n)
        hx_n = self.hnet_dense1(x_n)
        hx_n = nn.relu(hx_n)
        hx_n = self.hnet_dense2(hx_n)
        hx_n = nn.relu(hx_n)
        hx_n = self.hnet_dense3(hx_n)

        reward = r + shaping * (self.gamma * hx_n - hx)

        return reward
