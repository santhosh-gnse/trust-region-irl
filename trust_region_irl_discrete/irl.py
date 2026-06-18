import numpy as np
from itertools import product
from tqdm import tqdm
import math
import copy
import trajectory as T
import matplotlib.pyplot as plt
import plot as P
import optimizer as O


def linear_decay(lr0=0.2, decay_rate=1.0, decay_steps=1):
    def _lr(k):
        return lr0 / (1.0 + decay_rate * np.floor(k / decay_steps))
    return _lr

def no_op(lr0=0.2):
    def _lr(k):
        return lr0
    return _lr

def logsumexp(x, axis=0):
    x_max = np.max(x, axis=axis, keepdims=True)
    return (x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))).squeeze()


# Local environment using p_transition
class GridEnv:
    def __init__(self, terminal_set, p_initial, p_transition):
        self.state = 0
        self.terminal_set = terminal_set
        self.p_initial = p_initial
        self.p_transition = p_transition
        self.n_states, _, self.n_actions = p_transition.shape

    def reset(self):
        self.state = np.random.choice(list(range(len(self.p_initial))), p=self.p_initial)
        return self.state

    def step(self, action, reward=None):
        probs = self.p_transition[self.state, :, action]
        next_state = np.random.choice(self.n_states, p=probs)
        if reward is not None:
            r = reward[self.state, action]
        else:
            r = None
        done = next_state in self.terminal_set
        self.state = next_state
        return next_state, r, done

def expected_rho_from_trajectories(n_states, n_actions, features, trajectories):
    """
    Compute the feature expectation of the given trajectories.
    """
    n_state_action_pairs, n_features = features.shape

    transitions = np.concatenate([np.array(trj.transitions()) for trj in trajectories], axis=0)
    transitions = transitions[:,0:2]

    s_a_space = []
    for s in range(n_states):
        for a in range(n_actions):
            s_a_space.append([s, a])
    s_a_space = np.array(s_a_space)
    vec_to_idx = {tuple(row): idx for idx, row in enumerate(s_a_space)}

    counted_transitions = {s_a_idx: 0 for s_a_idx in range(n_state_action_pairs)}
    for t in transitions:
        idx = vec_to_idx[tuple(t)]
        counted_transitions[idx] += 1

    # Get avg occupancy
    rho = np.array(list(counted_transitions.values()))
    rho = rho/len(trajectories)

    # Normalize to ensure that rho is a probability distribution
    rho = rho / rho.sum()
    return rho.flatten()


def expected_svf_from_trajectories(n_states, features, trajectories, absorbing_state=24, normalize=True):
    features = np.asarray(features)
    assert features.shape == (n_states, n_states)

    transitions = np.concatenate([np.asarray(trj.transitions()) for trj in trajectories], axis=0)
    if transitions.ndim == 1:
        transitions = transitions[None, :]

    # count visited states s_t
    s = transitions[:, 0].astype(int)
    assert (s >= 0).all() and (s < n_states).all()
    counts = np.bincount(s, minlength=n_states).astype(np.float64)

    # additionally count absorbing when it appears as next-state s_{t+1}
    if transitions.shape[1] >= 3:
        sp = transitions[:, 2].astype(int)
        # only count the absorbing hits
        counts[absorbing_state] += np.sum(sp == absorbing_state)

    # average per-trajectory (matches your earlier convention)
    rho_s = counts / max(len(trajectories), 1)

    if normalize:
        total = rho_s.sum()
        if total > 0:
            rho_s /= total

    return rho_s


def initial_probabilities_from_trajectories(n_states, trajectories):
    """
    Compute the probability of a state being a starting state using the
    given trajectories.
    """
    p = np.zeros(n_states)

    for t in trajectories:
        p[t.transitions()[0][0]] += 1.0

    p = p / len(trajectories)
    p = p.clip(1e-6, 1.0)
    p = p / p.sum()

    return p


def expected_policy_rho_by_rollouts(p_transition, p_initial, terminal, p_action, features, absorbing, num_episodes=100, max_steps_per_episode=20, **kwargs):
    """
    Compute the expected state visitation frequency using the given local
    action probabilities by rolling out the policy. 
    """
    num_episodes = kwargs.get("rollouts_num_episodes", num_episodes)
    max_steps_per_episode = kwargs.get("rollouts_max_steps_per_episode", max_steps_per_episode)
    n_states, _, n_actions = p_transition.shape

    terminal_set = set(list(terminal.nonzero()[0])).union(set(list(absorbing.nonzero()[0])))
    env = GridEnv(terminal_set, p_initial, p_transition)
    trajectories = []

    for _ in range(num_episodes):
        state = env.reset()

        traj = []
        for _ in range(max_steps_per_episode):
            action_probs = p_action[state]
            action = np.random.choice(n_actions, p=action_probs)

            next_state, _, done = env.step(action)
            traj.append((state, action, next_state))

            if done:
                break
            state = next_state
        
        trajectories.append(T.Trajectory(traj))
    
    svf_rollouts = expected_rho_from_trajectories(n_states, n_actions, features, trajectories)
    return svf_rollouts


def expected_svf_from_policy(p_transition, p_initial, terminal, p_action, absorbing, eps=1e-5):
    """
    Compute the expected state visitation frequency using the given local
    action probabilities.
    """
    n_states, _, n_actions = p_transition.shape
    final = list(terminal.nonzero()[0]) + list(absorbing.nonzero()[0])
    p_transition = np.copy(p_transition)
    p_transition[final, :, :] = 0.0

    # set-up transition matrices for each action
    p_transition = [np.array(p_transition[:, :, a]) for a in range(n_actions)]

    # actual forward-computation of state expectations
    d = np.zeros(n_states)

    delta = np.inf
    while delta > eps:
        d_ = [p_transition[a].T.dot(p_action[:, a] * d) for a in range(n_actions)]
        d_ = p_initial + np.array(d_).sum(axis=0)

        delta, d = np.max(np.abs(d_ - d)), d_

    rho = d / d.sum()
    return rho.flatten()



def expected_rho_from_policy(p_transition, p_initial, terminal, p_action, absorbing, eps=1e-5):
    """
    Compute the expected state visitation frequency using the given local
    action probabilities.
    """
    n_states, _, n_actions = p_transition.shape
    final = list(terminal.nonzero()[0]) + list(absorbing.nonzero()[0])
    p_transition = np.copy(p_transition)
    p_transition[final, :, :] = 0.0

    # set-up transition matrices for each action
    p_transition = [np.array(p_transition[:, :, a]) for a in range(n_actions)]

    # actual forward-computation of state expectations
    d = np.zeros(n_states)

    delta = np.inf
    while delta > eps:
        d_ = [p_transition[a].T.dot(p_action[:, a] * d) for a in range(n_actions)]
        d_ = p_initial + np.array(d_).sum(axis=0)
        d_ = (d_/d_.sum()).clip(1e-3, 1.0)

        delta, d = np.max(np.abs(d_ - d)), d_

    rho = p_action * d[:, np.newaxis]
    rho = rho / rho.sum()

    return rho.flatten()


def soft_policy_evaluation(p_transition, pi, reward, discount, terminal=None, absorbing=None, alpha=1.0, eps=1e-5):
    """
    n_statesoft (entropy-regularized) policy evaluation

      Q_pi^soft(s,a) = E_{s'~P(.|s,a)}[ R + gamma V_pi^soft(s') ]
      V_pi^soft(s)   = E_{a~pi(.|s)}[ Q_pi^soft(s,a) - alpha log pi(a|s) ]
    """
    n_states, _, n_actions = p_transition.shape
    pi = pi.reshape(n_states, n_actions)
    reward = reward.reshape(n_states, n_actions)

    if absorbing is None:
        absorbing = np.zeros(n_states, dtype=bool)
    if terminal is None:
        terminal = np.zeros(n_states, dtype=bool)
    terminal_set = terminal | absorbing

    # transitions probabilities
    p = [np.array(p_transition[:, :, a]) for a in range(n_actions)]
    r = [reward[:, a] for a in range(n_actions)]

    # entropy term: -alpha log pi(a|s)
    log_pi = np.log(np.clip(pi, 1e-10, 1.0))
    ent = -alpha * log_pi  # (n_states,n_actions)

    v = np.zeros(n_states)
    delta = np.inf

    while delta > eps:
        v_old = v.copy()

        # clamp bootstrap on terminals
        v_boot = v_old.copy()
        v_boot[terminal_set] = 0.0

        # Q(s,a) = E_{s'}[ R + gamma V(s') ]
        q = np.array([
            p[a] @ (r[a] + discount * v_boot)
            for a in range(n_actions)
        ]).T

        # V(s) = E_{a~pi}[ Q(s,a) - alpha log pi(a|s) ]
        v = (pi * (q + ent)).sum(axis=1)

        # hard clamp terminals
        v[terminal_set] = 0.0

        delta = np.max(np.abs(v - v_old))

    return v



def soft_value_iteration(p_transition, terminal, reward, discount, absorbing=None, pi_old=None, eta=None, do_RL=False, eps=1e-5, alpha=1.0, **kwargs):
    n_states, _, n_actions = p_transition.shape
    reward = reward.reshape(n_states, n_actions)

    if kwargs["trirl"]:
        reward = (1/(1 + eta))*(reward + eta * np.log(pi_old))

    if absorbing is None:
        absorbing = np.zeros(n_states, dtype=bool)
    if terminal is None:
        terminal = np.zeros(terminal, dtype=bool)
    
    terminal_set = terminal + absorbing
    terminal_entropy = -(1/n_actions)*(1/(1 - discount))*np.log(1/n_actions)

    # set up transition probability matrices
    p = [np.array(p_transition[:, :, a]) for a in range(n_actions)]
    reward = [reward[:, a] for a in range(n_actions)]

    # compute state log partition V and state-action log partition Q
    v = -1e200 * np.ones(n_states)  # np.dot doesn't behave with -np.inf

    delta = np.inf
    while delta > eps:
        v_old = v.copy()

        if do_RL:
            q = np.array([
                p[a] @ (reward[a] + discount * v_old)
                for a in range(n_actions)
            ]).T
        else:
            q = np.array([
                p[a] @ (reward[a] + (1 - terminal_set) * discount * v_old + terminal_set*(discount/(1-discount))*((1/kwargs["beta"])*terminal_entropy + reward[a]))
                for a in range(n_actions)
            ]).T

        if do_RL:
            v = alpha * logsumexp(q / alpha, axis=1)
        else:
            v = logsumexp(q, axis=1)

        delta = np.max(np.abs(v_old - v))

    if do_RL:
        pi = np.exp((q - v[:, None]) / alpha)
    else:
        pi = np.exp(q - v[:, None])

    logging_kl = None
    if kwargs["trirl"]:
        # computing KL(pi || pi_old) solely for logging. The kl minimization is done already by adding -log(pi_old) to the rewards
        logging_kl = np.mean((pi.clip(1e-10, 1.0) * (np.log(pi.clip(1e-10, 1.0)) - np.log(pi_old.clip(1e-10, 1.0)))).sum(axis=-1))

    return v, pi, logging_kl


def reinforcement_learning(p_transition, p_initial, terminal, reward, discount, features, pi_old=None, absorbing=None, eta=None,
                                eps_vi=1e-5, eps_rho=1e-5, **kwargs):

    # Soft Value Iteration
    v, p_action, logging_kl = soft_value_iteration(p_transition, terminal, reward, discount, absorbing, pi_old, eta, eps_vi, **kwargs)
    rho = expected_svf_from_policy(p_transition, p_initial, terminal, p_action, absorbing, eps_rho)

    return rho, p_action.clip(1e-10, 1.0), v, logging_kl

def trirl_reward_update(theta, rho_expert, rho_agent, features, epsilon=0.1, beta=0.5):
    theta_tilde = (1/features.sum(axis=-1))[:, None]*beta*(np.log(np.clip(rho_expert, 1e-6, 1.0)) - np.log(np.clip(rho_agent, 1e-6, 1.0)))
    theta_new = (1 - epsilon)*theta + epsilon*theta_tilde
    return theta_new, theta_tilde

def trirl_discrete(world, features, trajectories, expert_policy, init, irl_cfg,
               eps=1e-4, eps_vi=1e-5, eps_rho=1e-5):
    """
    Args:
        world: GridWorld object
        features: The feature-matrix maapping states to features
        trajectories: A list of `Trajectory` instances representing the
            expert demonstrations.
        expert_policy: expert policy
        irl_cfg: config
        eps: The threshold to be used as convergence criterion for the
            reward parameters.
        eps_lap: The threshold to be used as convergence criterion for the
            state partition function.
        eps_svf: The threshold to be used as convergence criterion for the
            expected state-visitation frequency.
    """

    def generator():
        max_iter = irl_cfg["max_iter"]
        count = 0
        while delta > eps and count < max_iter:
        # while count < max_iter:
            yield count
            count += 1

    p_transition = world.p_transition
    absorbing = world.absorbing.astype(bool)
    terminal = world.terminal.astype(bool)
    discount = irl_cfg["discount"]
    irl_cfg.pop("discount")
    n_states, _, n_actions = p_transition.shape

    # compute static properties
    # p_initial = initial_probabilities_from_trajectories(n_states, trajectories)
    p_initial = np.zeros((n_states))
    p_initial[world.initial] = 1.0 / len(world.initial)
    e_rho = expected_svf_from_policy(p_transition, p_initial, terminal, expert_policy, absorbing, eps_rho)
    # e_rho = expected_svf_from_trajectories(n_states, features, trajectories, absorbing_state=world.absorbing.nonzero()[0])
    e_features = features * e_rho[:, np.newaxis]

    # parameters
    theta = init(features.shape)
    delta = np.inf
    eta = lambda x: x

    # logging
    kl_pi_piold = []
    kl_rho_pi_rho_es = []
    value_fns = []
    log_rhoE_minus_r_betas = []
    reward_vecs = []
    rho_pis = []

    # initialize iterants
    # eta = linear_decay(lr0=irl_cfg["init_eta"], decay_rate=0.5)
    eta = no_op(lr0=irl_cfg["init_eta"])
    reward = np.sum(theta * features, axis=1)
    p_action = np.full((n_states, n_actions), 1/n_actions)
    rho = expected_svf_from_policy(p_transition, p_initial, terminal, p_action, absorbing, eps_rho)
    rho_agent = features * rho[:, np.newaxis]

    with tqdm(generator()) as pbar:
        for count in pbar:
            pbar.set_description(f"Delta: {delta} | eps: {eps}")

            # reset iterants
            theta_old = theta.copy()
            rho_agent_old = rho_agent.copy()
            pi_old = p_action.copy()

            # intermediate, large-stepped reward
            theta_intermediate, _ = trirl_reward_update(theta_old, e_features, rho_agent_old, features, epsilon=irl_cfg["epsilon"], beta=irl_cfg["beta"])
            reward_vec = np.sum(theta_intermediate * features, axis=1)  # r(s) or r(s,a) vector
            reward = reward_vec
            reward = np.repeat(reward[:, None], n_actions, axis=1)

            # tr policy update
            rho, p_action, value_fn, kl = reinforcement_learning(p_transition, p_initial, terminal, reward, discount, features, pi_old=pi_old, absorbing=absorbing, eta=eta(count),
                                                eps_vi=eps_vi, eps_rho=eps_rho, **irl_cfg)
            features_closed_form = features * rho[:, np.newaxis]

            # reward projection
            theta, theta_tilde = trirl_reward_update(theta_old, e_features, rho_agent_old, features, epsilon=irl_cfg["epsilon"]/(1 + eta(count)), beta=irl_cfg["beta"])
            rho_agent = features_closed_form

            # logging
            kl_rho_pi_rho_e = np.sum(rho_agent * (np.log(rho_agent.clip(1e-10, 1.0)) - np.log(e_features.clip(1e-10, 1.0))))
            kl_rho_pi_rho_es.append(kl_rho_pi_rho_e)
            kl_pi_piold.append(kl)

            # Eval Dual
            if irl_cfg["eval_dual"]:
                reward_projected = reward_vec.copy()
                reward_projected = np.repeat(reward_projected[:, None], n_actions, axis=1)
                # v_mce, _, _ = soft_value_iteration(
                #     p_transition, None, reward_projected, discount,
                #     trirl=False, absorbing=None, pi_old=None, eta=None,
                #     do_RL=True, eps=1e-5, alpha=1/irl_cfg["beta"])
                v_mce = soft_policy_evaluation(p_transition, copy.deepcopy(p_action), reward_projected, discount, terminal=None, absorbing=None, alpha=1/irl_cfg["beta"])
                value_fns.append(copy.deepcopy(v_mce))
                log_rhoE = np.log(e_rho.clip(eps_rho, 1.0))
                log_rhoE_minus_r_betas.append(log_rhoE - (reward_vec / irl_cfg["beta"]))
                reward_vecs.append(reward_vec.copy())
                rho_pis.append(rho.copy())

            # stopping criterion
            delta = np.max(np.abs(theta_old - theta))


    logs = {}
    logs["KL(pi || pi_old)"] = kl_pi_piold
    logs["KL(ρ_π || ρ_E)"] = kl_rho_pi_rho_es
    logs["value_fns"] = value_fns
    logs["log(ρ_E) - r/beta"] = log_rhoE_minus_r_betas
    logs["reward_vecs"] = reward_vecs
    logs["rho_pi"] = rho_pis
    logs["rho_E"] = e_rho

    return np.sum(theta * features, axis=1), p_action, logs
