"""Build the raw-state transition dataset for self-supervised encoder pretraining (pizza_mjx or pusht_mjx).

Sources (each transition keeps its source tag and episode id):
  expert        the expert demonstrations (--expert)
  random        smooth random actions (Ornstein-Uhlenbeck noise) from the env's start pose
  policy:<name> OPTIONAL (--params_dir) rollouts of earlier TRIRL checkpoints. Not used by default: they
                carry information from previous IRL runs into the representation.

Encoder inputs are the raw observation: pizza 17-D (tool pos rel home, arm qpos, arm qvel; the 18th
coverage channel is the task's success metric, so it is dropped), pushT the full 24-D observation.
No rewards or success labels are stored. Episodes are split 80/10/10 into train/val/test by
episode, per source.

The default pizza expert file is not in git (trirl_dataset/ is ignored), so it must exist locally;
otherwise pass an expert dataset with --expert (pushT: its expert .npz).

    python scripts/ssl/build_pizza_ssl_dataset.py --env pizza|pusht --out FILE.npz
"""
import argparse, glob, os, pickle
import numpy as np
import jax
import jax.numpy as jnp

from trust_region_irl.algorithms.trirl_ppo_fb.flax_full_jit.policy import Policy

# encoder inputs per env. pizza: tool pos (3) + arm qpos (7) + arm qvel (7); index 17 = coverage,
# the success metric, excluded. pusht: the full 24-D observation (block pose rel goal, ee rel goal,
# arm qpos, arm qvel), all raw state.
RAW = {"pizza": slice(0, 17), "pusht": slice(0, 24)}


def make_env(name, steps):
    if name == "pizza":
        from trust_region_irl.environments.pizza_mjx.environment import PizzaSpread
        return PizzaSpread(render=False, feature_fn="base", obs_mode="coverage", horizon=steps)
    from trust_region_irl.environments.pusht_mjx.environment import PushT
    return PushT(render=False, feature_fn="base", horizon=steps)


def rollout(env, act_fn, n_envs, n_steps, key):
    """Run n_envs envs for n_steps with auto-reset; return per-step arrays (T, N, ...)."""
    k_reset, k_run = jax.random.split(key)
    state = env.reset(jax.random.split(k_reset, n_envs), False)

    def step(carry, k):
        state, aux = carry
        obs = state.next_observation
        a, aux = act_fn(obs, aux, k)
        nstate = env.step(state, a)
        done = nstate.terminated | nstate.truncated
        return (nstate, aux), (obs, a, nstate.actual_next_observation, done, nstate.terminated)

    aux0 = jnp.zeros((n_envs, 2))
    _, out = jax.jit(lambda s, ks: jax.lax.scan(step, (s, aux0), ks))(state, jax.random.split(k_run, n_steps))
    return [np.asarray(o) for o in out]


def split_episodes(obs, act, nobs, done, term):
    """(T, N) streams -> list of episodes (obs, act, nobs, term). A trailing unfinished segment is kept
    as an episode too (it is a valid prefix; only its last step has no terminal flag)."""
    eps = []
    T, N = done.shape
    for i in range(N):
        start = 0
        for t in range(T):
            if done[t, i] or t == T - 1:
                sl = slice(start, t + 1)
                if t + 1 - start >= 2:
                    eps.append((obs[sl, i], act[sl, i], nobs[sl, i], term[sl, i]))
                start = t + 1
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="pizza", choices=["pizza", "pusht"])
    ap.add_argument("--expert", default="trirl_dataset/rl_expert/expert_dataset_pizza_synth_coverage.npz")
    ap.add_argument("--params_dir", default="",
                    help="optional: exported policies of earlier runs to add as a data source. Off by default: "
                         "those policies come from previous IRL runs, so using them leaks earlier results into "
                         "the representation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--random_envs", type=int, default=256)
    ap.add_argument("--policy_envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--ou_theta", type=float, default=0.15)
    ap.add_argument("--ou_sigma", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = make_env(args.env, args.steps)
    raw = RAW[args.env]
    key = jax.random.PRNGKey(args.seed)
    episodes = []   # (source, obs, act, nobs, term)

    # expert
    with np.load(args.expert) as f:   # read each array once (npz decompresses on every access)
        S, A, NS, AB = (f[k] for k in ("states", "actions", "next_states", "absorbing"))
        # episode boundaries: stored lengths, or (pusht) the absorbing flag on each episode's last step
        L = f["episode_lengths"] if "episode_lengths" in f.files else np.diff(np.r_[-1, np.where(AB > 0.5)[0]])
    ends = np.cumsum(L); starts = ends - L
    for s, e in zip(starts, ends):
        term = np.zeros(e - s, bool); term[-1] = bool(AB[e - 1])
        episodes.append(("expert", S[s:e], A[s:e], NS[s:e], term))
    print(f"expert: {len(L)} episodes, {int(L.sum())} transitions", flush=True)

    # random: OU noise gives smooth, wandering motion (iid noise barely moves the tool)
    def ou_act(obs, x, k):
        x = x + args.ou_theta * (0.0 - x) + args.ou_sigma * jax.random.normal(k, x.shape)
        return jnp.clip(x, -1.0, 1.0), x
    key, k = jax.random.split(key)
    eps = split_episodes(*rollout(env, ou_act, args.random_envs, args.steps, k))
    episodes += [("random",) + e for e in eps]
    print(f"random: {len(eps)} episodes, {sum(len(e[0]) for e in eps)} transitions", flush=True)

    # intermediate TRIRL policies, sampled stochastically with their own learned std
    pol = Policy(env.single_action_space.shape, 0.4, jnp.arange(env.single_observation_space.shape[0]))
    for f in (sorted(glob.glob(f"{args.params_dir}/*.pkl")) if args.params_dir else []):
        name = os.path.basename(f)[:-4]
        params = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(f, "rb"))["params"])

        def pol_act(obs, aux, k, params=params):
            m, ls = pol.apply(params, obs)
            return jnp.clip(m + jnp.exp(ls) * jax.random.normal(k, m.shape), -1.0, 1.0), aux
        key, k = jax.random.split(key)
        eps = split_episodes(*rollout(env, pol_act, args.policy_envs, args.steps, k))
        episodes += [(f"policy:{name}",) + e for e in eps]
        print(f"policy:{name}: {len(eps)} episodes, {sum(len(e[0]) for e in eps)} transitions", flush=True)

    # trajectory-level split per source
    rng = np.random.default_rng(args.seed)
    split = np.zeros(len(episodes), np.int8)          # 0 train, 1 val, 2 test
    sources = np.array([e[0] for e in episodes])
    for src in np.unique(sources):
        idx = rng.permutation(np.where(sources == src)[0])
        n = len(idx); split[idx[int(0.8 * n):int(0.9 * n)]] = 1; split[idx[int(0.9 * n):]] = 2

    src_names = list(np.unique(sources))
    cat = lambda j: np.concatenate([e[j] for e in episodes])
    lens = np.array([len(e[1]) for e in episodes])
    out = dict(
        obs=cat(1)[:, raw].astype(np.float32), actions=cat(2).astype(np.float32),
        next_obs=cat(3)[:, raw].astype(np.float32), terminated=cat(4).astype(bool),
        episode_id=np.repeat(np.arange(len(episodes)), lens).astype(np.int32),
        t_in_episode=np.concatenate([np.arange(n) for n in lens]).astype(np.int32),
        episode_lengths=lens.astype(np.int32), episode_split=split,
        episode_source=np.array([src_names.index(s) for s in sources], np.int16),
        source_names=np.array(src_names), env=np.array(args.env),
        dt=np.float32(1.0 / env.CONTROL_FREQ),
    )
    if args.env == "pizza":
        out.update(obs_fields=np.array(["tool_pos_rel_home(3)", "arm_qpos(7)", "arm_qvel(7)"]),
                   pizza_off=np.asarray(env.PIZZA_OFF, np.float32), pizza_r=np.float32(env.PIZZA_R))
    else:
        out.update(obs_fields=np.array(["block_pos_rel_goal(3)", "block_quat_rel_goal(4)", "ee_rel_goal(3)",
                                        "arm_qpos(7)", "arm_qvel(7)"]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"\nsaved {args.out}: {len(episodes)} episodes, {len(out['obs'])} transitions")
    for s in src_names:
        m = sources == s
        print(f"  {s:40s} episodes {m.sum():5d}  transitions {lens[m].sum():7d}  "
              f"(train {lens[m & (split == 0)].sum()}, val {lens[m & (split == 1)].sum()}, test {lens[m & (split == 2)].sum()})")


if __name__ == "__main__":
    main()
