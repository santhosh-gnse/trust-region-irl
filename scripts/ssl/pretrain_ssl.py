"""Self-supervised pretraining of a state encoder f(s) -> R^d for linear TRIRL reward features.

Method:
  spr   SPR (Schwarzer et al. 2021): action-conditioned K-step latent prediction, EMA target, cosine loss

The reward features are the output of the frozen online encoder. No rewards or success labels are
used. Inputs are the raw observations (17-D pizza, 24-D pushT).

    python scripts/ssl/pretrain_ssl.py --data D.npz --method spr --out DIR
"""
import argparse, json, os, pickle, time
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


# ---------------------------------------------------------------- networks
from trust_region_irl.representation.encoder import Encoder   # shared with TRIRL's frozen features


class MLP(nn.Module):
    out: int
    hidden: int = 256

    @nn.compact
    def __call__(self, x):
        x = nn.gelu(nn.LayerNorm()(nn.Dense(self.hidden)(x)))
        return nn.Dense(self.out)(x)


def cos_loss(p, t):
    p = p / (jnp.linalg.norm(p, axis=-1, keepdims=True) + 1e-8)
    t = t / (jnp.linalg.norm(t, axis=-1, keepdims=True) + 1e-8)
    return 1.0 - jnp.sum(p * t, axis=-1)


# ---------------------------------------------------------------- data
class Data:
    """Episode-aware access: future state at offset k and the action sequence in between,
    never crossing an episode boundary."""

    def __init__(self, path, split):
        d = np.load(path)
        ep, lens = d["episode_id"], d["episode_lengths"]
        keep = d["episode_split"][ep] == split
        self.obs = jnp.asarray(d["obs"][keep]); self.act = jnp.asarray(d["actions"][keep])
        self.nobs = jnp.asarray(d["next_obs"][keep])
        epk = ep[keep]
        # end index (exclusive) of each transition's episode, in the filtered arrays
        _, first, counts = np.unique(epk, return_index=True, return_counts=True)
        end = np.zeros(len(epk), np.int64)
        for f, c in zip(first, counts):
            end[f:f + c] = f + c
        self.end = jnp.asarray(end)
        self.n = len(epk)
        self.source = d["episode_source"][epk]
        self.source_names = list(d["source_names"])

    def tree_flatten(self):
        return (self.obs, self.act, self.nobs, self.end), self.n

    @classmethod
    def tree_unflatten(cls, n, leaves):
        d = cls.__new__(cls)
        d.obs, d.act, d.nobs, d.end = leaves
        d.n = n
        return d

    def future(self, i, k):
        """State k steps after transition i (k >= 1), and whether it exists in the same episode."""
        j = i + k
        e = self.end[i]
        s = jnp.where((j < e)[:, None], self.obs[jnp.minimum(j, self.n - 1)], self.nobs[e - 1])
        return s, j <= e

    def actions(self, i, K):
        """a_{i..i+K-1} (zero-padded past the episode end) and a per-step validity mask."""
        idx = i[:, None] + jnp.arange(K)[None]
        valid = idx < self.end[i][:, None]
        return jnp.where(valid[..., None], self.act[jnp.minimum(idx, self.n - 1)], 0.0), valid


jax.tree_util.register_pytree_node(Data, Data.tree_flatten, Data.tree_unflatten)


# ---------------------------------------------------------------- methods
def build(method, d, obs_dim, act_dim, K, key):
    if method != "spr":
        raise ValueError(method)
    enc = Encoder(d)
    ks = jax.random.split(key, 4)
    P = {"enc": enc.init(ks[0], jnp.zeros((1, obs_dim)))}
    trans, proj, pred = MLP(d), MLP(64), MLP(64)
    P["trans"] = trans.init(ks[1], jnp.zeros((1, d + act_dim)))
    P["proj"] = proj.init(ks[2], jnp.zeros((1, d)))
    P["pred"] = pred.init(ks[3], jnp.zeros((1, 64)))
    target = {"enc": P["enc"], "proj": P["proj"]}

    def loss_fn(P, target, data, i, key):
        acts, valid = data.actions(i, K)
        zh, total, msum = enc.apply(P["enc"], data.obs[i]), 0.0, 0.0
        for k in range(1, K + 1):
            zh = trans.apply(P["trans"], jnp.concatenate([zh, acts[:, k - 1]], -1))
            s_k, ok = data.future(i, k)
            tgt = proj.apply(target["proj"], enc.apply(target["enc"], s_k))
            y = pred.apply(P["pred"], proj.apply(P["proj"], zh))
            m = (ok & valid[:, k - 1]).astype(jnp.float32)
            total += jnp.sum(m * cos_loss(y, jax.lax.stop_gradient(tgt))); msum += m.sum()
        loss = total / jnp.maximum(msum, 1.0)
        return loss, {"pred": loss}

    feat = lambda P, s: enc.apply(P["enc"], s)
    return P, target, loss_fn, feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--method", default="spr", choices=["spr"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--K", type=int, default=5, help="SPR prediction depth")
    ap.add_argument("--tau", type=float, default=0.99, help="SPR EMA")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tr, va = Data(args.data, 0), Data(args.data, 1)
    mu = tr.obs.mean(0); sd = tr.obs.std(0) + 1e-6
    for D in (tr, va):   # encoder input standardisation, fixed from the train split
        D.obs = (D.obs - mu) / sd; D.nobs = (D.nobs - mu) / sd
    key = jax.random.PRNGKey(args.seed)
    key, kb = jax.random.split(key)
    P, target, loss_fn, feat = build(args.method, args.d, tr.obs.shape[1], tr.act.shape[1], args.K, kb)

    log = []
    t0 = time.time()
    if loss_fn is not None:
        opt = optax.adam(args.lr)
        opt_state = opt.init(P)

        # the data is passed in as an argument, not closed over, so the arrays are not baked into the graph
        @jax.jit
        def train_step(P, target, opt_state, data, key):
            ki, kl = jax.random.split(key)
            i = jax.random.randint(ki, (args.batch,), 0, data.n)
            (loss, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(P, target, data, i, kl)
            upd, opt_state = opt.update(g, opt_state, P)
            P = optax.apply_updates(P, upd)
            target = jax.tree_util.tree_map(lambda t, o: args.tau * t + (1 - args.tau) * o,
                                            target, {"enc": P["enc"], "proj": P["proj"]})
            return P, target, opt_state, loss, aux

        @jax.jit
        def val_batch(P, target, data, key):
            ki, kl = jax.random.split(key)
            i = jax.random.randint(ki, (args.batch,), 0, data.n)
            return loss_fn(P, target, data, i, kl)


        def val_loss(P, target, key):
            outs = [val_batch(P, target, va, k) for k in jax.random.split(key, 4)]
            return (np.mean([float(o[0]) for o in outs]),
                    {a: np.mean([float(o[1][a]) for o in outs]) for a in outs[0][1]})

        for step in range(1, args.steps + 1):
            key, k = jax.random.split(key)
            P, target, opt_state, loss, aux = train_step(P, target, opt_state, tr, k)
            if step % 1000 == 0 or step == 1:
                key, k = jax.random.split(key)
                vl, vaux = val_loss(P, target, k)
                row = {"step": step, "train": float(loss), "val": float(vl),
                       **{f"val_{a}": float(b) for a, b in vaux.items()}}
                log.append(row)
                print(json.dumps(row), flush=True)

    # feature statistics on the train split (for standardise + clip at TRIRL time)
    f_all = jnp.concatenate([feat(P, tr.obs[j:j + 65536]) for j in range(0, tr.n, 65536)])
    f_mu, f_sd = np.asarray(f_all.mean(0)), np.asarray(f_all.std(0) + 1e-6)
    zq = (np.asarray(f_all) - f_mu) / f_sd
    f_lo, f_hi = np.percentile(zq, 0.5, axis=0), np.percentile(zq, 99.5, axis=0)
    feat_params = {"enc": P["enc"]}
    with open(f"{args.out}/encoder.pkl", "wb") as fh:
        pickle.dump({"method": args.method, "args": vars(args), "d": args.d,
                     "feature_params": jax.tree_util.tree_map(np.asarray, feat_params),
                     "feature_key": list(feat_params)[0],
                     "obs_mean": np.asarray(mu), "obs_std": np.asarray(sd),
                     "feat_mean": f_mu, "feat_std": f_sd, "feat_lo": f_lo, "feat_hi": f_hi}, fh)
    json.dump({"log": log, "seconds": time.time() - t0, "args": vars(args)}, open(f"{args.out}/train_log.json", "w"))
    print(f"saved {args.out}/encoder.pkl  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
