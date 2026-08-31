#!/bin/sh

# Do not let JAX pre-grab the GPU. Without this it takes 75% of the card up
# front (~60 GB of an 80 GB device) regardless of what the run actually needs,
# which blocks everyone else sharing the machine.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# TRIRL max eta on Ant
python experiment.py \
    --algorithm.name="trirl_trpl.flax_full_jit" \
    --algorithm.data_path="../trirl_dataset/rl_expert/Ant-v5_30_PPO.npz" \
    --algorithm.total_timesteps=30e6 \
    --environment.name="ant_mjx" \
    --environment.nr_envs=4096 \
    --environment.seed=0 \
    --runner.mode="train" \
    --runner.track_tb=True \
    --runner.track_console=True \
    --runner.track_wandb=False \
    --runner.save_model=False \
    --runner.wandb_entity="your-wandb-entity" \
    --runner.project_name="trust_region_irl" \
    --runner.exp_name="ant_trirl" \



# TRIRL PPO FB
python experiment.py \
   --algorithm.name="trirl_ppo_fb.flax_full_jit" \
   --algorithm.data_path="../trirl_dataset/rl_expert/expert_dataset_pusht_mtp_50_episodes_trirl_f32abs.npz" \
   --algorithm.total_timesteps=100e6 \
   --algorithm.entropy_coef=0.001 \
   --algorithm.clip_range=0.2 \
   --algorithm.env_reward_frac=0.0 \
   --algorithm.nr_steps=128 \
   --algorithm.nr_epochs=10 \
   --algorithm.nr_epochs_disc=10 \
   --algorithm.minibatch_size=512 \
   --algorithm.learning_rate_disc=1e-04 \
   --algorithm.learning_rate=4e-04 \
   --algorithm.epsilon=0.2 \
   --algorithm.init_eta=30.0 \
   --algorithm.gp_lambda=0.5 \
   --algorithm.gae_lambda=0.95 \
   --algorithm.std_dev=0.4 \
   --algorithm.evaluation_and_save_frequency=2097152 \
   --environment.name="pusht_mjx" \
   --environment.nr_envs=4096 \
   --environment.seed=0 \
   --runner.mode="train" \
   --runner.track_tb=True \
   --runner.track_console=True \
   --runner.track_wandb=True \
   --runner.save_model=True \
   --runner.wandb_entity="trirl" \
   --runner.project_name="role_ip" \
   --runner.exp_name="pusht_ppo_fb"


# TRIRL PPO FB - RBF Feature
python experiment.py \
   --algorithm.name="trirl_ppo_fb.flax_full_jit" \
   --algorithm.data_path="../trirl_dataset/rl_expert/expert_dataset_pusht_mtp_50_episodes_trirl_f32abs.npz" \
   --algorithm.total_timesteps=100e6 \
   --algorithm.entropy_coef=0.001 \
   --algorithm.clip_range=0.2 \
   --algorithm.env_reward_frac=0.0 \
   --algorithm.nr_steps=128 \
   --algorithm.nr_epochs=10 \
   --algorithm.nr_epochs_disc=10 \
   --algorithm.minibatch_size=512 \
   --algorithm.learning_rate_disc=1e-04 \
   --algorithm.learning_rate=4e-04 \
   --algorithm.epsilon=0.2 \
   --algorithm.init_eta=30.0 \
   --algorithm.gp_lambda=0.5 \
   --algorithm.gae_lambda=0.95 \
   --algorithm.std_dev=0.4 \
   --algorithm.evaluation_and_save_frequency=2097152 \
   --environment.feature_fn=base_rbf \
   --environment.name="pusht_mjx" \
   --environment.nr_envs=4096 \
   --environment.seed=0 \
   --runner.mode="train" \
   --runner.track_tb=True \
   --runner.track_console=True \
   --runner.track_wandb=True \
   --runner.save_model=True \
   --runner.wandb_entity="trirl" \
   --runner.project_name="role_ip" \
   --runner.exp_name="pusht_ppo_fb"


# TRIRL PPO FB - RBF Feature with new cleaned dataset
python experiment.py \
   --algorithm.name="trirl_ppo_fb.flax_full_jit" \
   --algorithm.data_path="../trirl_dataset/rl_expert/expert_dataset_pusht_mtp_clean_93_episodes_trirl_f32abs.npz" \
   --algorithm.total_timesteps=100e6 \
   --algorithm.entropy_coef=0.001 \
   --algorithm.clip_range=0.2 \
   --algorithm.env_reward_frac=0.0 \
   --algorithm.nr_steps=128 \
   --algorithm.nr_epochs=10 \
   --algorithm.nr_epochs_disc=10 \
   --algorithm.minibatch_size=512 \
   --algorithm.learning_rate_disc=1e-04 \
   --algorithm.learning_rate=4e-04 \
   --algorithm.anneal_learning_rate=True \
   --algorithm.epsilon=0.2 \
   --algorithm.init_eta=30.0 \
   --algorithm.gp_lambda=0.5 \
   --algorithm.gae_lambda=0.95 \
   --algorithm.std_dev=0.4 \
   --algorithm.evaluation_and_save_frequency=2097152 \
   --environment.feature_fn="base_rbf" \
   --environment.name="pusht_mjx" \
   --environment.nr_envs=4096 \
   --environment.seed=0 \
   --runner.mode="train" \
   --runner.track_tb=True \
   --runner.track_console=True \
   --runner.track_wandb=True \
   --runner.save_model=True \
   --runner.wandb_entity="trirl" \
   --runner.project_name="role_ip" \
   --runner.exp_name="pusht_ppo_fb"


# TRIRL PPO FB - Boltzmann Feature-based on a cleaner Data with RBF
python experiment.py \
   --algorithm.name="trirl_ppo_fb.flax_full_jit" \
   --algorithm.data_path="../trirl_dataset/rl_expert/expert_dataset_pusht_mtp_clean_93_episodes_trirl_f32abs.npz" \
   --algorithm.total_timesteps=150e6 \
   --algorithm.entropy_coef=0.001 \
   --algorithm.clip_range=0.2 \
   --algorithm.env_reward_frac=0.0 \
   --algorithm.nr_steps=128 \
   --algorithm.nr_epochs=10 \
   --algorithm.nr_epochs_disc=10 \
   --algorithm.minibatch_size=512 \
   --algorithm.learning_rate_disc=1e-04 \
   --algorithm.learning_rate=4e-04 \
   --algorithm.reward_type=boltzmann-feature-based \
   --algorithm.dsm_alpha=0.001 \
   --algorithm.dsm_sigma=0.5 \
   --algorithm.epsilon=0.2 \
   --algorithm.init_eta=30.0 \
   --algorithm.gp_lambda=0.5 \
   --algorithm.gae_lambda=0.95 \
   --algorithm.std_dev=0.4 \
   --algorithm.anneal_learning_rate=True \
   --algorithm.evaluation_and_save_frequency=2097152 \
   --environment.name="pusht_mjx" \
   --environment.nr_envs=4096 \
   --environment.seed=0 \
   --environment.feature_fn="base_rbf" \
   --runner.mode="train" \
   --runner.track_tb=True \
   --runner.track_console=True \
   --runner.track_wandb=True \
   --runner.save_model=True \
   --runner.wandb_entity="trirl" \
   --runner.project_name="role_ip" \
   --runner.exp_name="pusht_ppo_fb"



# TRIRL PPO FB - Boltzmann Feature-based
python experiment.py \
   --algorithm.name="trirl_ppo_fb.flax_full_jit" \
   --algorithm.data_path="../trirl_dataset/rl_expert/expert_dataset_pusht_mtp_50_episodes_trirl_f32abs.npz" \
   --algorithm.total_timesteps=100e6 \
   --algorithm.entropy_coef=0.001 \
   --algorithm.clip_range=0.2 \
   --algorithm.env_reward_frac=0.0 \
   --algorithm.nr_steps=128 \
   --algorithm.nr_epochs=10 \
   --algorithm.nr_epochs_disc=10 \
   --algorithm.minibatch_size=512 \
   --algorithm.learning_rate_disc=1e-04 \
   --algorithm.learning_rate=4e-04 \
   --algorithm.reward_type=boltzmann-feature-based \
   --algorithm.dsm_alpha=0.001 \
   --algorithm.dsm_sigma=0.5 \
   --algorithm.epsilon=0.2 \
   --algorithm.init_eta=30.0 \
   --algorithm.gp_lambda=0.5 \
   --algorithm.gae_lambda=0.95 \
   --algorithm.std_dev=0.4 \
   --algorithm.evaluation_and_save_frequency=2097152 \
   --environment.name="pusht_mjx" \
   --environment.nr_envs=4096 \
   --environment.seed=0 \
   --runner.mode="train" \
   --runner.track_tb=True \
   --runner.track_console=True \
   --runner.track_wandb=True \
   --runner.save_model=True \
   --runner.wandb_entity="trirl" \
   --runner.project_name="role_ip" \
   --runner.exp_name="pusht_ppo_fb"

# TRIRL PPO-FB on the 3dof Push-T (block_type=3dof, 120-episode 3dof expert data)
python experiment.py \
   --algorithm.name="trirl_ppo_fb.flax_full_jit" \
   --algorithm.data_path="../trirl_dataset/rl_expert/expert_dataset_pusht_mtp_3dof_120_episodes_trirl_f32abs.npz" \
   --algorithm.total_timesteps=200e6 \
   --algorithm.entropy_coef=0.001 \
   --algorithm.clip_range=0.2 \
   --algorithm.env_reward_frac=0.0 \
   --algorithm.nr_steps=128 \
   --algorithm.nr_epochs=10 \
   --algorithm.nr_epochs_disc=10 \
   --algorithm.minibatch_size=512 \
   --algorithm.learning_rate_disc=1e-04 \
   --algorithm.learning_rate=4e-04 \
   --algorithm.reward_type=boltzmann-feature-based \
   --algorithm.dsm_alpha=0.001 \
   --algorithm.dsm_sigma=0.5 \
   --algorithm.epsilon=0.2 \
   --algorithm.init_eta=30.0 \
   --algorithm.gp_lambda=0.5 \
   --algorithm.gae_lambda=0.95 \
   --algorithm.std_dev=0.4 \
   --algorithm.anneal_learning_rate=True \
   --algorithm.evaluation_and_save_frequency=2097152 \
   --environment.name="pusht_mjx" \
   --environment.nr_envs=4096 \
   --environment.seed=0 \
   --environment.feature_fn="base_rbf" \
   --environment.block_type="3dof" \
   --runner.mode="train" \
   --runner.track_tb=True \
   --runner.track_console=True \
   --runner.track_wandb=True \
   --runner.save_model=True \
   --runner.wandb_entity="trirl" \
   --runner.project_name="role_ip" \
   --runner.exp_name="pusht_ppo_fb" \
   --runner.run_name="pusht_3dof_120"


# =============================================================================
# TRIRL PPO-FB on the bulb-screwing task -- first full run
#
# Expert: 27 REAL kinesthetic demonstrations (2026-08-30), relabelled at the
# ACHIEVED action scales (0.17 m/s, 2.6 rad/s) and truncated at screwed-home.
# 12,725 transitions, episode lengths 364-619.
#
# Every choice below is either carried from the pushT run that actually solved
# (runs/role_ip/pusht_ppo_fb/1782780263) or forced by a measurement on this task.
#
#   reward_type=boltzmann-feature-based
#       On pushT the LINEAR reward parked at 0.20-0.23 and never finished:
#       r=theta^T phi matches the expert's AVERAGE feature, so the policy
#       stabilises at the average distance. Only the Boltzmann (nonlinear)
#       reward solved it. Same failure mode applies here.
#
#   feature_fn=base_screw
#       The task's blind spot: base/base_rbf are yaw-invariant and use d_seat,
#       a 3-D norm, so they cannot tell RESTING PROUD (5.6 mm) from SCREWED
#       DOWN. base_screw splits that into d_xy + |depth| and adds a screwing
#       term (turning while engaged). Its seated_bump reads 0.03 when resting
#       vs 1.00 when home; base_rbf's seat_tight reads 0.92 vs 1.00 -- almost
#       no signal. This is the pushT "peaked features" lesson applied to the
#       part of THIS task that actually distinguishes success.
#
#   gamma=0.999
#       The task takes ~450 steps. At the default 0.99 the effective horizon is
#       100 steps, so success is discounted to ~0.01 and is invisible from the
#       episode start. 0.999 gives an effective horizon of 1000. (DPPO uses
#       exactly 0.999 for the FurnitureBench tasks, which are the same length.)
#
#   nr_epochs_disc=10
#       The solving pushT run used 10; later runs at 20 did not solve.
#
#   anneal_learning_rate=True + save_model=True
#       On pushT the solve existed only at ONE eval peak and the final
#       checkpoint missed it entirely. save-best is not optional here.
#
#   nr_envs=4096 with 12,725 expert transitions
#       Batch = 4096*128 = 524k >> the expert set, so the expert minibatch is
#       resampled WITH REPLACEMENT (the fixed sampler). Ratio is slightly better
#       than pushT's, which worked.
#
# Smoke-test first -- 5M steps is enough to see the loop is healthy:
#   ... --algorithm.total_timesteps=5e6 --runner.track_wandb=False
# =============================================================================
python experiment.py \
   --algorithm.name="trirl_ppo_fb.flax_full_jit" \
   --algorithm.data_path="../trirl_dataset/rl_expert/expert_dataset_bulbscrew_real_27.npz" \
   --algorithm.total_timesteps=200e6 \
   --algorithm.gamma=0.999 \
   --algorithm.entropy_coef=0.001 \
   --algorithm.clip_range=0.2 \
   --algorithm.env_reward_frac=0.0 \
   --algorithm.nr_steps=128 \
   --algorithm.nr_epochs=10 \
   --algorithm.nr_epochs_disc=10 \
   --algorithm.minibatch_size=512 \
   --algorithm.learning_rate_disc=1e-04 \
   --algorithm.learning_rate=4e-04 \
   --algorithm.reward_type=boltzmann-feature-based \
   --algorithm.dsm_alpha=0.001 \
   --algorithm.dsm_sigma=0.5 \
   --algorithm.epsilon=0.2 \
   --algorithm.init_eta=30.0 \
   --algorithm.gp_lambda=0.5 \
   --algorithm.gae_lambda=0.95 \
   --algorithm.std_dev=0.4 \
   --algorithm.anneal_learning_rate=True \
   --algorithm.evaluation_and_save_frequency=2097152 \
   --environment.name="bulbscrew_mjx" \
   --environment.nr_envs=4096 \
   --environment.seed=0 \
   --environment.feature_fn="base_screw" \
   --runner.mode="train" \
   --runner.track_tb=True \
   --runner.track_console=True \
   --runner.track_wandb=True \
   --runner.save_model=True \
   --runner.wandb_entity="trirl" \
   --runner.project_name="role_ip" \
   --runner.exp_name="bulbscrew_ppo_fb" \
   --runner.run_name="bulbscrew_real_27_bs"
