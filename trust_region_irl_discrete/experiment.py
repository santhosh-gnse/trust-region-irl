import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import gridworld as W
import irl as IRL
import plot as P
import trajectory as T
import solver as S
import optimizer as O


def logsumexp(x: np.ndarray) -> float:
    """
    Numerically stable logsumexp over a 1D array.
    """
    x = np.asarray(x)
    m = np.max(x)
    return float(m + np.log(np.sum(np.exp(x - m))))

def sym_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p)
    q = np.asarray(q)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    kl_pq = np.sum(p * (np.log(p + eps) - np.log(q + eps)))
    kl_qp = np.sum(q * (np.log(q + eps) - np.log(p + eps)))
    return float(kl_pq + kl_qp)


def parse_cfg(cfg: Union[DictConfig, dict]) -> Dict[str, Any]:
    def recurse(d, parent_key=""):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, (DictConfig, dict)):
                items.update(recurse(v, new_key))
            else:
                items[new_key] = v
        return items
    return recurse(cfg)

@hydra.main(version_base=None, config_path=".", config_name="config")
def experiment(cfg : DictConfig) -> None:

    # Save the config in the current working directory (automatically changed by Hydra)
    exp_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    with open(f"{exp_dir}/config.yaml", "w") as f:
        OmegaConf.save(config=cfg, f=f)

    env_cfg = parse_cfg(cfg.env)
    expert_cfg = parse_cfg(cfg.expert)
    plotting_cfg = cfg.plotting
    irl_cfg = parse_cfg(cfg.mce_irl)


    """ 
    Set up plot style 
    """

    plt.rcParams.update({
        "font.family": plotting_cfg["rc_params"]["font"],
        "axes.titlesize": plotting_cfg["rc_params"]["titlesize"],
        "axes.labelsize": plotting_cfg["rc_params"]["labelsize"],
        "legend.fontsize": plotting_cfg["rc_params"]["fontsize"],
        "xtick.labelsize": plotting_cfg["rc_params"]["ticksize"],
        "ytick.labelsize": plotting_cfg["rc_params"]["ticksize"],
    })

    nrows = 2
    ncols = 2
    base_figsize = tuple( plotting_cfg["rc_params"]["figsize"])
    total_figsize = (base_figsize[0] * 2, base_figsize[1] * 2)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=total_figsize, sharey=True, sharex=True, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.0, hspace=0.01)
    axes = axes.flatten()
    for ax in axes:
        ax.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)


    """
    Set-up our MDP/GridWorld
    """
    # set up absorbing states and terminal states
    absorbing = np.zeros(int(env_cfg["world_size"]**2), dtype=bool)
    absorbing[env_cfg["absorbing_idx"]] = True
    terminal = np.zeros(int(env_cfg["world_size"]**2), dtype=bool)
    terminal[env_cfg["terminal_idx"]] = True
    initial_states = env_cfg["initial_idx"]
    
    # create our world
    world = W.IcyGridWorld(size=env_cfg["world_size"], p_slip=env_cfg["p_slip"], absorbing=absorbing, terminal=terminal, initial=initial_states)

    # set up the reward function
    r_default, r_abs, r_goal = env_cfg["r_default"], env_cfg["r_abs"], env_cfg["r_goal"]
    penultimate_coords, rewarding_actions_idx = env_cfg["penultimate_coords"], env_cfg["rewarding_actions_idx"]
    reward = np.full((world.n_states, world.n_actions), r_default)
    reward[absorbing.nonzero(), :] = r_abs
    for coords, action_idx in zip(penultimate_coords, rewarding_actions_idx):
        row = coords[0]
        col = coords[1]
        n = world.size
        reward[int(row*n + col), action_idx] = r_goal


    ax = axes[0]
    state_only_reward = np.zeros((25))
    state_only_reward[world.absorbing.nonzero()[0]] = 1.0
    P.plot_state_values(ax, world, state_only_reward, **plotting_cfg["style"], cbar_off=plotting_cfg["rc_params"]["cbar_off"])
    ax.set_title("Expert Reward")


    """
    Generate expert trajectories
    """

    # generate trajectories
    value, expert_policy, _ = IRL.soft_value_iteration(world.p_transition, terminal, reward, irl_cfg["discount"], trirl=False, absorbing=None, pi_old=None, eta=None, do_RL=True, eps=1e-5)
    expert_policy_exec = T.stochastic_policy_adapter(expert_policy)
    expert_trajectories = list(T.generate_trajectories(expert_cfg["n_trajectories"], world, expert_policy_exec))

    """
    Trust Region Inverse Reinforcement Learning
    """
    
    # set up features
    features = W.state_features(world)
    init = O.Constant(irl_cfg["init_reward"])

    # trust region IRL
    reward_maxcausal, policy_maxcausal, logs = IRL.trirl_discrete(world, features, expert_trajectories, expert_policy, init, irl_cfg, eps=1e-3)

    # normalize
    reward_maxcausal = (reward_maxcausal - reward_maxcausal.min())/(reward_maxcausal.max() - reward_maxcausal.min())

    """
    Plot Results
    """

    # Plot other logs
    # for k, v in logs.items():
    #     else:
    #         ax = plt.figure(num=f"{k}").add_subplot(111)
    #         plt.title(k)
    #         if TABLE_TEXT:
    #             plt.figtext(0.5, 0.85, table_text,
    #                         ha='center', va='top',
    #                         fontsize=6, family='monospace',
    #                         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    #         plt.plot(np.array(v))
    #         plt.draw()

    # Grid of rewards and policies
    ax = axes[2]
    P.plot_stochastic_policy(ax, world, expert_policy, **plotting_cfg["style"], cbar_off=plotting_cfg["rc_params"]["cbar_off"])
    ax.set_title("Expert Policy")

    ax = axes[1]
    P.plot_state_values(ax, world, reward_maxcausal, **plotting_cfg["style"], cbar_off=plotting_cfg["rc_params"]["cbar_off"])
    ax.set_title("Learnt Reward")
    
    ax = axes[3]
    P.plot_stochastic_policy(ax, world, policy_maxcausal, **plotting_cfg["style"], cbar_off=plotting_cfg["rc_params"]["cbar_off"])
    ax.set_title("Learnt Policy")
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=plotting_cfg["style"]["cmap"])
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", fraction=0.04, pad=0.01)
    cbar.set_ticks([0, 1])
    fig.savefig("./reward_policy_comparison.pdf", dpi=plotting_cfg["rc_params"]["dpi"], bbox_inches='tight')

    # primal and dual
    kl_series = logs.get("KL(ρ_π || ρ_E)", None)
    if kl_series is not None:
        mpl.rcParams['text.usetex'] = False
        fig_kl, ax_kl = plt.subplots(figsize=(5,5))
        ax_kl.semilogy(np.array(kl_series), color='#1E90FF', lw=4)
        ax_kl.set_title(r"KL$(\rho_\pi \,\|\, \rho_{\rm E})$")
        ax_kl.set_xlabel(r"Iterations")
        fig_kl.savefig("kl_plot.pdf", dpi=plotting_cfg["rc_params"]["dpi"], bbox_inches='tight')
        plt.close(fig_kl)

    value_fns = logs.get("value_fns", None)
    log_rhoE_minus_r_betas = logs.get("log(ρ_E) - r/beta", None)
    if log_rhoE_minus_r_betas is not None:
        g_series = []
        for term, value_fn in zip(log_rhoE_minus_r_betas, value_fns):
            v_0 = value_fn[env_cfg["initial_idx"]]
            g = v_0 + irl_cfg["beta"] * logsumexp(term)
            g_series.append(g)

        fig_g, ax_g = plt.subplots(figsize=(5,5))
        ax_g.plot(np.array(g_series), color='#FF6F00', lw=4)
        ax_g.set_title(r"G(r_i)")
        ax_g.set_xlabel(r"Iterations")
        fig_g.savefig("G_plot.pdf", dpi=plotting_cfg["rc_params"]["dpi"], bbox_inches='tight')
        plt.close(fig_g)


if __name__ == "__main__":
    experiment()