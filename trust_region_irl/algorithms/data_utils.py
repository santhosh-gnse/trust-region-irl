import numpy as np
import jax
import jax.numpy as jnp

# def prepare_expert_data(data_path, cutoff=1):
#     dataset = dict()

#     # load expert training data
#     expert_files = np.load(data_path)
#     cutoff = int(len(expert_files["states"])/cutoff)

#     dataset["states"] = expert_files["states"][:cutoff]
#     dataset["actions"] = expert_files["actions"][:cutoff]

#     # maybe we have next action and next next state
#     try:
#         dataset["next_actions"] = expert_files["next_actions"][:cutoff]
#         dataset["next_next_states"] = expert_files["next_next_states"][:cutoff]
#     except KeyError as e:
#         print("Did not find next action or next next state.")

#     # maybe we have next states and dones in the dataset
#     try:
#         dataset["next_states"] = expert_files["next_states"][:cutoff].squeeze()
#         dataset["absorbing"] = (expert_files["absorbing"]).flatten()[:cutoff].squeeze()
#     except KeyError as e:
#         print("Warning Dataset: %s" % e)

#     # maybe we have episode returns, if yes done
#     try:
#         dataset["episode_returns"] = expert_files["episode_returns"][:cutoff].squeeze()
#         return dataset
#     except KeyError:
#         print("Warning Dataset: No episode returns. Falling back to step-based reward.")

#     # this has to work
#     try:
#         dataset["rewards"] = expert_files["rewards"][:cutoff].squeeze()
#         return dataset
#     except KeyError:
#         raise KeyError("The dataset has neither an episode nor a step-based reward!")


def prepare_expert_data(data_path, cutoff=1):
    dataset = dict()
    expert_files = np.load(data_path)

    def _flatten_feature_array(x):
        x = np.asarray(x)
        if x.ndim <= 2:
            return x
        return x.reshape(-1, x.shape[-1])   # (30,1000,d) -> (30000,d)

    def _flatten_scalar_array(x):
        return np.asarray(x).reshape(-1)     # (30,1000) -> (30000,)

    # load and normalize expert training data
    states = _flatten_feature_array(expert_files["states"])
    actions = _flatten_feature_array(expert_files["actions"])

    cutoff = int(states.shape[0] / cutoff)

    dataset["states"] = states[:cutoff]
    dataset["actions"] = actions[:cutoff]

    # maybe we have next action and next next state
    try:
        dataset["next_actions"] = _flatten_feature_array(expert_files["next_actions"])[:cutoff]
        dataset["next_next_states"] = _flatten_feature_array(expert_files["next_next_states"])[:cutoff]
    except KeyError:
        print("Did not find next action or next next state.")

    # maybe we have next states and dones in the dataset
    try:
        dataset["next_states"] = _flatten_feature_array(expert_files["next_states"])[:cutoff]
        dataset["absorbing"] = _flatten_scalar_array(expert_files["absorbing"])[:cutoff]
    except KeyError as e:
        print("Warning Dataset: %s" % e)

    # maybe we have episode returns, if yes done
    try:
        dataset["episode_returns"] = _flatten_scalar_array(expert_files["episode_returns"])[:cutoff]
        return dataset
    except KeyError:
        print("Warning Dataset: No episode returns. Falling back to step-based reward.")

    # this has to work
    try:
        dataset["rewards"] = _flatten_scalar_array(expert_files["rewards"])[:cutoff]
        return dataset
    except KeyError:
        raise KeyError("The dataset has neither an episode nor a step-based reward!")



def expert_data_spec(num_samples, state_dim, action_dim):
    """
    Dummy spec for use with full jittting
    """
    return {
        "states": jax.ShapeDtypeStruct(
            shape=(num_samples, state_dim),
            dtype=jnp.float32,
        ),
        "actions": jax.ShapeDtypeStruct(
            shape=(num_samples, action_dim),
            dtype=jnp.float32,
        ),
        "next_states": jax.ShapeDtypeStruct(
            shape=(num_samples, state_dim),
            dtype=jnp.float32,
        ),
        "absorbing": jax.ShapeDtypeStruct(
            shape=(num_samples,),
            dtype=jnp.float32,
        ),
        "rewards": jax.ShapeDtypeStruct(
            shape=(num_samples,),
            dtype=jnp.float32,
        ),
    }