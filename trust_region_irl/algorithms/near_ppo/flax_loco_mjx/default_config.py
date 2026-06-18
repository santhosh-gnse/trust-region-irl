from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name
    config.hidden_layers = [512, 256]
    config.lr = 1e-4
    config.num_envs = 2048
    config.num_steps = 10
    config.total_timesteps = 10e7
    config.update_epochs = 20
    config.num_minibatches = 64
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.clip_eps = 0.2
    config.init_std = 1.0
    config.learnable_std = True
    config.ent_coef = 0.001
    config.vf_coef = 1.0
    config.max_grad_norm = 10.0
    config.activation = "tanh"
    config.anneal_lr = False
    config.weight_decay = 0.0
    config.normalize_env = True
    config.debug = False
    config.n_seeds = 1 
    config.vmap_across_seeds = True
    config.validation_active = False
    config.validation_num_steps = 100
    config.validation_num_envs = 100
    config.validation_num = 10  # set to 0 to disable validation

    config.proportion_env_reward = 0.0

    config.batch_size_ncsn = 256
    config.minibatch_size_ncsn = 64
    config.total_samples_ncsn = 25e6
    config.nr_epochs_ncsn = 20 # Number of ncsn epochs
    config.anneal_power_ncsn = 2.0
    config.sigma_begin_ncsn = 10.0
    config.sigma_end_ncsn = 0.01
    config.L_ncsn = 20
    config.use_running_mean_stand = False
    config.hidden_layers_encoder_ncsn = [512, 1024, 2048, 4096]
    config.hidden_layers_decoder_ncsn = [2048, 1024, 512, 128, 32]
    config.ncsn_lr = 0.000118
    config.sigma_inference_ncsn = -1
    config.anneal_threshold = 0.03
    config.env_reward_frac = 0.0
    config.handle_absorbing_states = True
    config.state_based = True
    config.ncsnv1 = True
    config.data_path = "../trirl_dataset/rl_expert/MjxUnitreeGo2_30_PPO.npz"
    config.mocap_data_path = "../trirl_dataset/mocap"

    return config
