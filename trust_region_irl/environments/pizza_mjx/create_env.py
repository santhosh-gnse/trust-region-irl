from trust_region_irl.environments.pizza_mjx.environment import PizzaSpread
from trust_region_irl.environments.pizza_mjx.general_properties import GeneralProperties


def create_train_and_eval_env(config):
    train_env = PizzaSpread(render=config.environment.render, feature_fn=config.environment.feature_fn,
                        obs_mode=getattr(config.environment, "obs_mode", "plain"),
                        horizon=getattr(config.environment, "horizon", 500),
                        success_threshold=getattr(
                            config.environment, "success_threshold", 0.90),
                        feature_encoder=getattr(config.environment, "feature_encoder", ""))
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval:
        return train_env, train_env

    eval_env = PizzaSpread(render=config.environment.render, feature_fn=config.environment.feature_fn,
                        obs_mode=getattr(config.environment, "obs_mode", "plain"),
                        horizon=getattr(config.environment, "horizon", 500),
                        success_threshold=getattr(
                            config.environment, "success_threshold", 0.90),
                        feature_encoder=getattr(config.environment, "feature_encoder", ""))
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env
