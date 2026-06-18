import os
import argparse

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now you can import the module
from locomujoco_override import *

from omegaconf import OmegaConf
from pathlib import Path

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ')

# Set up argument parser
parser = argparse.ArgumentParser(description='Run evaluation with GAILJax.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
parser.add_argument('--goal_override', action='store_true', help='Ovveride random velocity targets to a +x velocity target')
args = parser.parse_args()

# Use the path from command line arguments
path = args.path
agent_conf, agent_state = PPOJax.load_agent(path)
config = agent_conf.config

if args.goal_override:
    config.experiment.env_params.goal_type = "GoalXRootVelocity"
    config.experiment.env_params.reward_type = "LocomotionRewardOverride"

    deterministic = True
    n_steps = 1000
    n_envs = 1
    record = False
    save_traj = True
    save_path = str(Path(path).parent) + "/"

# get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# create env
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = False
env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

# Determine which evaluation environment to run
if args.use_mujoco:
    # run eval mujoco
    PPOJax.play_policy_mujoco(env, agent_conf, agent_state, deterministic=deterministic, n_steps=n_steps, record=record,
                               train_state_seed=0)
else:
    # run eval mjx
    PPOJax.play_policy(env, agent_conf, agent_state, deterministic=deterministic, n_steps=n_steps, n_envs=n_envs, record=record,
                        train_state_seed=0, save_traj=save_traj, save_path=save_path)
