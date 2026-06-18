from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name
    config.hidden_layers = [512, 256]
    config.lr = 1.02e-05
    config.disc_lr = 0.000775
    config.num_steps = 20
    config.total_timesteps = 400e6
    config.update_epochs = 20
    config.disc_minibatch_size = 512
    config.proportion_env_reward = 0.0
    config.n_disc_epochs = 30
    config.num_minibatches = 64
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.clip_eps = 0.2
    config.init_std = 0.4
    config.learnable_std = True
    config.ent_coef = 4e-05
    config.disc_ent_coef = 0.001
    config.vf_coef = 1.0
    config.max_grad_norm = 1.0
    config.activation = "tanh"
    config.anneal_lr = False
    config.weight_decay = 0.0
    config.normalize_env = True
    config.debug = False
    config.n_seeds = 1
    config.vmap_across_seeds = True
    config.data_path = "../trirl_dataset/rl_expert/MjxUnitreeGo2_30_PPO.npz"
    config.mocap_data_path = "../trirl_dataset/mocap"

    config.validation_active = False
    config.validation_num_steps = 100
    config.validation_num_envs = 100
    config.validation_num = 10  # set to 0 to disable validation


    config.handle_absorbing_states = True
    config.gp_lambda = 0.16

    config.reward_type = 'state-based' # options: state-action, state-based, shaped
    config.reward_approximator_type = 'state-based' # options: state-action, state-based, shaped
    config.epsilon = 0.443
    config.disc_buffer_capacity = 100

    config.mean_bound = 0.000251
    config.cov_bound = 0.000182
    config.trust_region_coef = 0.828
    config.on_demand_etas = False

    config.beta = float(1/config.ent_coef)
    config.chunk_size = 4

    config.n_reward_approximator_epochs = 30
    config.reward_approximator_lr = 5e-05
    config.reward_fn_approximator = True
    config.warm_start = True

    return config

