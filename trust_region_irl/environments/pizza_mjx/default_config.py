from ml_collections import config_dict


def get_config(environment_name):
    config = config_dict.ConfigDict()

    config.name = environment_name

    config.seed = 1
    config.nr_envs = 64
    config.render = False
    config.device = "gpu"
    config.copy_train_env_for_eval = True
    config.feature_fn = "base"  # base | base_rbf | spread | state_action
    config.obs_mode = "plain"   # plain (17-D, matches the dataset) | coverage (18-D)
    # 500 covers the longest retimed demonstration (454 steps at tempo 0.85)
    config.horizon = 500
    config.success_threshold = 0.90     # swept-coverage fraction counted as solved

    return config
