from trust_region_irl.environments.loco_mjx.general_properties import GeneralProperties
from trust_region_irl.environments.loco_mjx.locomujoco_override import *
from loco_mujoco import RLFactory, ImitationFactory

ENV_PARAMS = {   
"MjxUnitreeGo2": {
    "env_name": "MjxUnitreeGo2",
    "horizon": 1000,
    "terminal_state_type": "HeightBasedTerminalStateHandler",
    "goal_type": "GoalXRootVelocity",  # GoalXRootVelocity or NoGoal or GoalRandomRootVelocity
    "goal_params": {},  # Empty as per original snippet
    "visualize_goal": False,
    "headless": True,
    "reward_type": "LocomotionRewardOverride",  # SafeTargetXVelocityReward OR SafeLocomotionReward OR LocomotionRewardOverride
    "reward_params": {
        "tracking_w_exp_xy": 4.0,
        "tracking_w_exp_yaw": 4.0,
        "tracking_w_sum_xy": 2.0,
        "tracking_w_sum_yaw": 1.0,
        "air_time_coeff": 0.1,
        "joint_acc_coeff": 2.0e-5,
        "air_time_max": 0.5,
        "joint_torque_coeff": 2.0e-7,
        "joint_position_limit_coeff": 2.0,
        "action_rate_coeff": 0.1,
        "symmetry_air_coeff": 0.005,
        "energy_coeff": 5.0e-5
    },
},

"MjxUnitreeG1": {
        "env_name": "MjxUnitreeG1",
        "horizon": 1000,
        "terminal_state_type": "HeightBasedTerminalStateHandler",
        "goal_type": "NoGoal",
        "goal_params": {
            "visualize_goal": False,
        },
        "headless": True,
        "reward_type": "SafeTargetXVelocityReward",
        "reward_params": {
            "target_velocity": 2.0,
        },
    }
}


ENV_PARAMS_MOCAP = {
"MjxUnitreeG1": {
    "env_name": "MjxUnitreeG1",
    "headless": True,
    "disable_arms": False,
    "horizon": 1000,
    "goal_type": "GoalTrajMimic",
    "reward_type": "MimicReward",
    "reward_params": {
        "qpos_w_sum": 0.0,
        "qvel_w_sum": 0.0,
        "rpos_w_sum": 0.5,
        "rquat_w_sum": 0.3,
        "rvel_w_sum": 0.1,
        "sites_for_mimic": [
            "upper_body_mimic",
            "left_hand_mimic",
            "left_foot_mimic",
            "right_hand_mimic",
            "right_foot_mimic"
        ]
    }
},

"MjxUnitreeH1": {
    "env_name": "MjxUnitreeH1",
    "headless": True,
    "disable_arms": False,
    "horizon": 1000,
    "goal_type": "GoalTrajMimic",
    "reward_type": "MimicReward",
    "reward_params": {
        "qpos_w_sum": 0.0,
        "qvel_w_sum": 0.0,
        "rpos_w_sum": 0.5,
        "rquat_w_sum": 0.3,
        "rvel_w_sum": 0.1,
        "sites_for_mimic": [
            "upper_body_mimic",
            "left_hand_mimic",
            "left_foot_mimic",
            "right_hand_mimic",
            "right_foot_mimic"
        ]
    }
}
}



def create_env(config):
    assert "task" in config.environment, "Please mention a task: either rl or a mocap dataset name"
    if config.environment.task == "rl":
        PARAMS = ENV_PARAMS
        factory = RLFactory
        task_params = {}
    elif config.environment.task in ["walk", "run", "balance", "squat"]:
        PARAMS = ENV_PARAMS_MOCAP
        factory = ImitationFactory
        task_params = {"default_dataset_conf": {"task": config.environment.task}}
    else:
        raise NotImplementedError

    assert "agent" in config.environment, f"Please choose an agent name for the loco_mjx environment. Available Tasks: {list(PARAMS.keys())}"

    env_params = PARAMS[config.environment.agent]
    if config.runner.mode=="test":
        env_params["headless"] = False

    env = factory.make(**env_params, **task_params)
    env.close = lambda: None

    env.general_properties = GeneralProperties

    return env, env
