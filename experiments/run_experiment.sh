#!/bin/sh

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