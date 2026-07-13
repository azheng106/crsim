"""PPO agent for MiniClash

Why PPO here? Diagnostics showed the Q-learning agents get stuck in an elixir-dumping
local optimum: they spend on cheap troops as fast as elixir regenerates and never
discover the "hold to 10, commit a Giant + support" counterpush. That trajectory needs
~15+ coordinated decisions in a row, which epsilon-greedy (random single-action jitter)
essentially never samples -- so the payoff of holding is never experienced. PPO explores
with a stochastic policy shaped by an entropy bonus, which samples coherent multi-step
behaviour far more readily, and its clipped objective keeps self-play updates stable.

Architecture: a single shared actor-critic (two hidden ReLU layers, then a policy head
over the 17 actions and a scalar value head). Because the observation is
perspective-normalized, one network plays both sides -- every P0 and P1 step is training
data, and self-play is literally the policy against a copy of itself.
"""

import numpy as np

from agent import N_FEATURES, N_ACTIONS

_NEG = -1e9  # stand-in for -inf on masked logits (keeps softmax finite)


class PPOAgent:
    def __init__(self, hidden: int = 128, lr: float = 3e-4, clip: float = 0.2,
                 c_value: float = 0.5, c_entropy: float = 0.01, seed: int | None = None):
        rng = np.random.default_rng(seed)
        H, F, A = hidden, N_FEATURES, N_ACTIONS

        def he(out_dim, in_dim):
            return (rng.standard_normal((out_dim, in_dim)) * np.sqrt(2.0 / in_dim)).astype(np.float32)

        self.W1, self.b1 = he(H, F), np.zeros(H, np.float32)
        self.W2, self.b2 = he(H, H), np.zeros(H, np.float32)
        self.Wp, self.bp = (he(A, H) * 0.01), np.zeros(A, np.float32)  # small init -> near-uniform start
        self.Wv, self.bv = he(1, H), np.zeros(1, np.float32)
        self._params = ["W1", "b1", "W2", "b2", "Wp", "bp", "Wv", "bv"]

        self.lr, self.clip, self.c_value, self.c_entropy = lr, clip, c_value, c_entropy
        self.rng = rng
        self._adam = {p: (np.zeros_like(getattr(self, p)), np.zeros_like(getattr(self, p)))
                      for p in self._params}
        self._adam_t = 0
        self.eps = 0.0  # unused (kept so training code can set it uniformly)

    # --- forward ---
    def _forward(self, X: np.ndarray):
        z1 = X @ self.W1.T + self.b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.W2.T + self.b2
        h2 = np.maximum(z2, 0.0)
        logits = h2 @ self.Wp.T + self.bp
        value = (h2 @ self.Wv.T + self.bv)[:, 0]
        cache = (X, z1, h1, z2, h2)
        return logits, value, cache

    @staticmethod
    def _masked_log_softmax(logits, mask):
        z = np.where(mask, logits, _NEG)
        z = z - z.max(axis=1, keepdims=True)
        logsumexp = np.log(np.exp(z).sum(axis=1, keepdims=True))
        return z - logsumexp  # (B, A)

    # --- inference API (compatible with eval/watch in train.py) ---
    def policy(self, phi: np.ndarray, mask: np.ndarray):
        logits, value, _ = self._forward(phi[None, :])
        logp = self._masked_log_softmax(logits, mask[None, :])[0]
        return logp, float(value[0])

    def act_collect(self, phi: np.ndarray, mask: np.ndarray):
        """Sample an action from the masked policy. Returns (action, logp, value)."""
        logp, value = self.policy(phi, mask)
        p = np.exp(logp)
        p = p / p.sum()
        a = int(self.rng.choice(N_ACTIONS, p=p))
        return a, float(logp[a]), value

    def select_action(self, phi: np.ndarray, mask: np.ndarray, greedy: bool = False) -> int:
        logp, _ = self.policy(phi, mask)
        if greedy:
            return int(np.argmax(np.where(mask, logp, _NEG)))
        p = np.exp(np.where(mask, logp, _NEG))
        p = p / p.sum()
        return int(self.rng.choice(N_ACTIONS, p=p))

    # --- loss + gradients for one minibatch ---
    def loss_and_grads(self, X, mask, actions, old_logp, adv, ret):
        B = X.shape[0]
        logits, value, (X_, z1, h1, z2, h2) = self._forward(X)
        logp_all = self._masked_log_softmax(logits, mask)          # (B, A)
        p = np.exp(logp_all)                                       # (B, A), masked ~ 0
        idx = np.arange(B)
        logp = logp_all[idx, actions]                             # (B,)

        ratio = np.exp(logp - old_logp)
        surr1 = ratio * adv
        surr2 = np.clip(ratio, 1 - self.clip, 1 + self.clip) * adv
        unclipped = surr1 <= surr2  # min() picks surr1; only then does gradient flow
        policy_loss = -np.mean(np.minimum(surr1, surr2))

        value_loss = 0.5 * np.mean((value - ret) ** 2)
        entropy = -(p * np.where(mask, logp_all, 0.0)).sum(axis=1)  # (B,)
        entropy_loss = -self.c_entropy * np.mean(entropy)
        total = policy_loss + self.c_value * value_loss + entropy_loss

        # --- backward ---
        # d policy_loss / d logp  (per sample); d logp/d logits_j = onehot - p
        gp = np.where(unclipped, -adv * ratio, 0.0) / B            # (B,)
        onehot = np.zeros_like(p); onehot[idx, actions] = 1.0
        dlogits = gp[:, None] * (onehot - p)                       # (B, A)
        # entropy: d entropy_loss / d logits_j = c_e/B * p_j (logp_j + entropy)
        dlogits += (self.c_entropy / B) * p * (np.where(mask, logp_all, 0.0) + entropy[:, None])
        # value head
        dv = self.c_value * (value - ret) / B                     # (B,)

        dWp = dlogits.T @ h2; dbp = dlogits.sum(axis=0)
        dWv = (dv[None, :] @ h2); dbv = np.array([dv.sum()], np.float32)
        dh2 = dlogits @ self.Wp + dv[:, None] @ self.Wv
        dz2 = dh2 * (z2 > 0)
        dW2 = dz2.T @ h1; db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.W2
        dz1 = dh1 * (z1 > 0)
        dW1 = dz1.T @ X_; db1 = dz1.sum(axis=0)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
                 "Wp": dWp, "bp": dbp, "Wv": dWv.astype(np.float32), "bv": dbv}
        info = {"total": float(total), "policy": float(policy_loss),
                "value": float(value_loss), "entropy": float(entropy.mean())}
        return grads, info

    def _adam_step(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self._adam_t += 1
        for p, g in grads.items():
            m, v = self._adam[p]
            m[:] = beta1 * m + (1 - beta1) * g
            v[:] = beta2 * v + (1 - beta2) * (g * g)
            mhat = m / (1 - beta1 ** self._adam_t)
            vhat = v / (1 - beta2 ** self._adam_t)
            getattr(self, p)[...] -= self.lr * mhat / (np.sqrt(vhat) + eps)

    def update(self, X, mask, actions, old_logp, adv, ret, epochs=4, minibatch=256):
        """Run K epochs of clipped-surrogate minibatch SGD over a collected rollout."""
        N = X.shape[0]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # normalize advantages
        last = {}
        for _ in range(epochs):
            order = self.rng.permutation(N)
            for s in range(0, N, minibatch):
                b = order[s:s + minibatch]
                grads, last = self.loss_and_grads(X[b], mask[b], actions[b],
                                                   old_logp[b], adv[b], ret[b])
                self._adam_step(grads)
        # Report entropy only over real decision states (>1 legal action). Most ticks
        # only no-op is legal (elixir too low), which forces entropy to ~0 and would
        # otherwise swamp the mean -- useless for monitoring exploration health.
        decision = mask.sum(axis=1) > 1
        if decision.any():
            logits, _, _ = self._forward(X[decision])
            lp = self._masked_log_softmax(logits, mask[decision])
            p = np.exp(lp)
            last["entropy_dec"] = float((-(p * np.where(mask[decision], lp, 0.0)).sum(1)).mean())
        return last

    # --- persistence ---
    def save(self, path: str):
        np.savez(path, **{p: getattr(self, p) for p in self._params})

    @classmethod
    def load(cls, path: str) -> "PPOAgent":
        d = np.load(path)
        agent = cls(hidden=d["b1"].shape[0])
        for p in agent._params:
            setattr(agent, p, d[p])
        return agent


def compute_gae(rewards, values, boot_value, gamma=0.997, lam=0.95):
    """GAE for a single trajectory. `boot_value` bootstraps a truncated (non-terminal)
    episode; pass 0.0 when the episode ended in a real win/loss. Returns (adv, returns)."""
    T = len(rewards)
    adv = np.zeros(T, np.float32)
    nextval, nextadv = boot_value, 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * nextval - values[t]
        adv[t] = delta + gamma * lam * nextadv
        nextval, nextadv = values[t], adv[t]
    returns = adv + np.asarray(values, np.float32)
    return adv, returns


def grad_check(seed=0, tol=1e-3):
    """Numerically verify loss_and_grads against finite differences on a random batch."""
    rng = np.random.default_rng(seed)
    ag = PPOAgent(hidden=16, seed=seed)
    # Finite-difference checks need float64: the tiny changes small gradients produce are
    # below float32's resolution and would spuriously round to 0.
    for p in ag._params:
        setattr(ag, p, getattr(ag, p).astype(np.float64))
    B = 8
    X = rng.standard_normal((B, N_FEATURES))
    mask = rng.random((B, N_ACTIONS)) > 0.3
    mask[:, 0] = True  # keep at least one legal action per row
    actions = np.array([int(rng.choice(np.flatnonzero(mask[i]))) for i in range(B)])
    old_logp = rng.standard_normal(B) * 0.1
    adv = rng.standard_normal(B)
    ret = rng.standard_normal(B)

    grads, _ = ag.loss_and_grads(X, mask, actions, old_logp, adv, ret)

    def loss_only():
        _, info = ag.loss_and_grads(X, mask, actions, old_logp, adv, ret)
        return info["total"]

    worst = 0.0
    for name in ["W1", "b1", "W2", "b2", "Wp", "bp", "Wv", "bv"]:
        P = getattr(ag, name)
        flat = P.ravel()
        g = grads[name].ravel()
        for k in rng.choice(flat.size, size=min(8, flat.size), replace=False):
            orig = flat[k]; h = 1e-4
            flat[k] = orig + h; lp = loss_only()
            flat[k] = orig - h; lm = loss_only()
            flat[k] = orig
            num = (lp - lm) / (2 * h)
            # Combined absolute/relative error: relative error is meaningless when both
            # gradients are ~0 (finite differences round to 0 below their resolution).
            err = abs(num - g[k])
            if abs(num) + abs(g[k]) > 1e-4:
                err /= abs(num) + abs(g[k])
            worst = max(worst, err)
    print(f"grad_check: worst error = {worst:.2e} ({'PASS' if worst < tol else 'FAIL'})")
    return worst < tol


if __name__ == "__main__":
    grad_check()
