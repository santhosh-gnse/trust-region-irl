from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name
    config.total_timesteps = 300e6
    config.lr = 5e-4
    config.anneal_lr = False
    config.buffer_size = 10e6
    config.learning_starts = 20000
    config.batch_size = 16384
    config.tau = 0.001
    config.gamma = 0.99
    config.target_entropy = "auto"
    config.log_std_min = -20
    config.log_std_max = 2
    config.hidden_layers = [512, 256]
    config.max_grad_norm = 0.5
    config.activation = "tanh"
    config.normalize_env = True
    config.logging_frequency = -1
    config.weight_decay = 0.0
    config.data_path = "../trirl_dataset/rl_expert/MjxUnitreeGo2_30_PPO.npz"
    config.mocap_data_path = "../trirl_dataset/mocap"

    config.init_std = 0.4
    config.learnable_std = True
    config.reward_type = 'state-based' # options: state-action, state-based, shaped
    config.init_ent_coeff = 0.01
    config.learn_ent_coeff = False
    config.gp_lambda = 0.0
    config.reg_mult = 1/(4*0.5)
    config.v0_loss = False
    config.use_lsiq = True
    config.nr_q_updates_per_step = 4
    config.use_target_q = True
    config.state_only = False


    return config
