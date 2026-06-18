from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.nr_parallel_seeds = 1
    config.total_timesteps = 50e6
    config.learning_rate = 5e-4
    config.anneal_learning_rate = False
    config.buffer_size = 5e6
    config.learning_starts = 10000
    config.batch_size = 8192
    config.tau = 0.005
    config.gamma = 0.99
    config.target_entropy = "auto"
    config.log_std_min = -20
    config.log_std_max = 2
    config.logging_frequency = 40960
    config.evaluation_and_save_frequency = -1  # -1 to disable
    config.evaluation_active = True
    config.evaluation_episodes = 10
    config.max_grad_norm = 0.5

    config.init_ent_coeff = 0.001
    config.learn_ent_coeff = False
    config.gp_lambda = 0.0
    config.reg_mult = 1/(4*0.5)
    config.v0_loss = False
    config.use_lsiq = True
    config.data_path = "../trirl_dataset/rl_expert/Ant-v5_30_PPO.npz"
    config.nr_q_updates_per_step = 12
    config.use_target_q = True

    return config
