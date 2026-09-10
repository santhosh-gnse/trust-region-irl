"""Mixed continuous (Gaussian) + discrete (Bernoulli) action distribution.

Some environments need one or more action dimensions to be a genuinely binary
decision (e.g. a latched gripper button) rather than a continuous value
thresholded at zero. Continuous-plus-threshold means the exploration noise has
to cross an arbitrary boundary every step -- a mean near zero is close to a
coinflip -- which is much noisier and slower to learn than a calibrated
Bernoulli probability.

The policy network's output shape is UNCHANGED. For a discrete dimension, the
`mean` slot is reinterpreted as a LOGIT (mean > 0 <=> P(action=+1) > 0.5), and
its `logstd` slot is simply unused (near-zero gradient, stays close to init).
That is what keeps this change local: Policy/get_policy never need to know
discrete dims exist, only trirl_ppo.py's sampling/loss code does.

`discrete_mask`: bool array, shape (action_dim,), True at indices that are
Bernoulli-in-{-1,+1} rather than Gaussian. Every function here reduces to
EXACTLY the original Gaussian-only formulas when discrete_mask is all False --
verified in scripts/verify_mixed_action_distribution.py, run before this is
wired into any live training.
"""
import jax
import jax.numpy as jnp


def sample(mean, logstd, key, discrete_mask):
    std = jnp.exp(logstd)
    key_g, key_b = jax.random.split(key)
    gauss = mean + std * jax.random.normal(key_g, shape=mean.shape)
    p = jax.nn.sigmoid(mean)
    bern = jnp.where(jax.random.uniform(key_b, shape=mean.shape) < p, 1.0, -1.0)
    return jnp.where(discrete_mask, bern, gauss)


def log_prob(action, mean, logstd, discrete_mask):
    std = jnp.exp(logstd)
    gauss_lp = -0.5 * ((action - mean) / std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - logstd
    bern_lp = jnp.where(action > 0, jax.nn.log_sigmoid(mean), jax.nn.log_sigmoid(-mean))
    return jnp.where(discrete_mask, bern_lp, gauss_lp)


def entropy(mean, logstd, discrete_mask):
    gauss_h = logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
    p = jax.nn.sigmoid(mean)
    bern_h = -p * jax.nn.log_sigmoid(mean) - (1.0 - p) * jax.nn.log_sigmoid(-mean)
    return jnp.where(discrete_mask, bern_h, gauss_h)


def kl(old_mean, old_logstd, mean, logstd, discrete_mask):
    """Per-dim analytic KL(old || new). Sum over axis=1 for the total, same as
    the original tr_loss_maha + tr_loss_cov_part decomposition it replaces."""
    old_std = jnp.exp(old_logstd)
    std = jnp.exp(logstd)
    gauss_kl = (0.5 * ((old_mean - mean) / old_std) ** 2
                + 0.5 * (2.0 * (jnp.log(old_std) - jnp.log(std)) + (std / old_std) ** 2 - 1.0))
    old_p = jax.nn.sigmoid(old_mean)
    bern_kl = (old_p * (jax.nn.log_sigmoid(old_mean) - jax.nn.log_sigmoid(mean))
              + (1.0 - old_p) * (jax.nn.log_sigmoid(-old_mean) - jax.nn.log_sigmoid(-mean)))
    return jnp.where(discrete_mask, bern_kl, gauss_kl)
