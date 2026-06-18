import jax.numpy as jnp

from trust_region_irl.algorithms.gail_ppo.flax_loco_mjx.gail_ppo import GAIL_PPO


class AMP_PPO(GAIL_PPO):

    @classmethod
    def _predict_rewards(cls, inputs, discriminator, disc_train_state):
        logits, _ = discriminator.apply({'params': disc_train_state.params,
                                         'run_stats': disc_train_state.run_stats},
                                        *inputs, mutable=["run_stats"])

        reward = jnp.maximum(0.0, 1 - 0.25*jnp.square(logits - 1))

        return reward

    @classmethod
    def _discriminator_loss(cls, config, logits, targets):

        # least squares loss
        total_loss = jnp.mean(jnp.square(logits - targets))

        return total_loss, logits

    @classmethod
    def _get_discriminator_targets(cls, plcy_batch_size, expert_batch_size):
        plcy_target = -1 * jnp.ones(shape=(plcy_batch_size,))
        expert_target = 1 * jnp.ones(shape=(expert_batch_size,))
        return plcy_target, expert_target
