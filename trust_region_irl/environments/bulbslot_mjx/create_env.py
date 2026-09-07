from trust_region_irl.environments.bulbslot_mjx.environment import BulbSlot
from trust_region_irl.environments.bulbslot_mjx.general_properties import GeneralProperties


def create_train_and_eval_env(config):
    train_env = BulbSlot(render=config.environment.render, feature_fn=config.environment.feature_fn)
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval:
        return train_env, train_env

    eval_env = BulbSlot(render=config.environment.render, feature_fn=config.environment.feature_fn)
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env
