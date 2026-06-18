import jax.numpy as jnp

from trust_region_irl.algorithms.gail_ppo.flax_loco_mjx.gail_ppo import GAIL_PPO


class AIRL_PPO(GAIL_PPO):

    @classmethod
    def _predict_rewards(cls, inputs, discriminator, disc_train_state):
        logits, _ = discriminator.apply({'params': disc_train_state.params,
                                         'run_stats': disc_train_state.run_stats},
                                        *inputs, mutable=["run_stats"])

        return logits
