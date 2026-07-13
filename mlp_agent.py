from collections import deque

import numpy as np

from agent import N_FEATURES, N_ACTIONS


class MLPQAgent:
    def __init__(self, hidden: int = 96, lr: float = 1e-3, td_clip: float = 1.0,
                 buffer_size: int = 50_000, batch_size: int = 128,
                 target_sync: int = 500, seed: int | None = None):
        rng = np.random.default_rng(seed)
        # He initialization for the ReLU layer.
        self.W1 = (rng.standard_normal((hidden, N_FEATURES)) * np.sqrt(2.0 / N_FEATURES)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (rng.standard_normal((N_ACTIONS, hidden)) * np.sqrt(2.0 / hidden)).astype(np.float32)
        self.b2 = np.zeros(N_ACTIONS, dtype=np.float32)
        self._params = ["W1", "b1", "W2", "b2"]
        self._sync_target()

        self.lr = lr
        self.td_clip = td_clip
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.eps = 1.0
        self.rng = rng

        self.buffer: deque = deque(maxlen=buffer_size)
        self._adam = {p: (np.zeros_like(getattr(self, p)), np.zeros_like(getattr(self, p)))
                      for p in self._params}
        self._t = 0

    # --- inference ---
    def _forward(self, Phi: np.ndarray, target: bool = False):
        W1, b1, W2, b2 = (
            (self.tW1, self.tb1, self.tW2, self.tb2) if target
            else (self.W1, self.b1, self.W2, self.b2)
        )
        z1 = Phi @ W1.T + b1
        a1 = np.maximum(z1, 0.0)
        q = a1 @ W2.T + b2
        return q, z1, a1

    def q_values(self, phi: np.ndarray) -> np.ndarray:
        q, _, _ = self._forward(phi[None, :])
        return q[0]

    def select_action(self, phi: np.ndarray, mask: np.ndarray, greedy: bool = False) -> int:
        legal = np.flatnonzero(mask)
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.choice(legal))
        q = np.where(mask, self.q_values(phi), -np.inf)
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    # --- training ---
    def remember(self, phi, action, nstep_return, phi_n, bootstrap, mask_n) -> None:
        """Store one n-step transition. `bootstrap` is gamma**n (0 if the episode ended
        within the window, so no value is bootstrapped)."""
        self.buffer.append((phi.astype(np.float32), action, np.float32(nstep_return),
                            phi_n.astype(np.float32), np.float32(bootstrap),
                            mask_n.astype(bool)))

    def learn(self) -> None:
        if len(self.buffer) < self.batch_size:
            return
        idx = self.rng.integers(0, len(self.buffer), size=self.batch_size)
        batch = [self.buffer[i] for i in idx]
        Phi = np.stack([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        G = np.array([b[2] for b in batch], dtype=np.float32)
        Phi_n = np.stack([b[3] for b in batch])
        boot = np.array([b[4] for b in batch], dtype=np.float32)
        Mask_n = np.stack([b[5] for b in batch])

        # n-step target: G + gamma**n * max_a' Q_target(phi_n, a') over legal actions.
        q_next, _, _ = self._forward(Phi_n, target=True)
        q_next = np.where(Mask_n, q_next, -np.inf)
        targets = G + boot * q_next.max(axis=1)

        q, z1, a1 = self._forward(Phi)
        B = self.batch_size
        pred = q[np.arange(B), actions]
        err = np.clip(pred - targets, -self.td_clip, self.td_clip)

        dq = np.zeros_like(q)
        dq[np.arange(B), actions] = err / B
        dW2 = dq.T @ a1
        db2 = dq.sum(axis=0)
        da1 = dq @ self.W2
        dz1 = da1 * (z1 > 0)
        dW1 = dz1.T @ Phi
        db1 = dz1.sum(axis=0)

        self._adam_step({"W1": dW1, "b1": db1, "W2": dW2, "b2": db2})

        self._t += 1
        if self._t % self.target_sync == 0:
            self._sync_target()

    def _adam_step(self, grads: dict, beta1=0.9, beta2=0.999, eps=1e-8) -> None:
        self._adam_t = getattr(self, "_adam_t", 0) + 1
        for p, g in grads.items():
            m, v = self._adam[p]
            m[:] = beta1 * m + (1 - beta1) * g
            v[:] = beta2 * v + (1 - beta2) * (g * g)
            mhat = m / (1 - beta1 ** self._adam_t)
            vhat = v / (1 - beta2 ** self._adam_t)
            getattr(self, p)[...] -= self.lr * mhat / (np.sqrt(vhat) + eps)

    def _sync_target(self) -> None:
        self.tW1, self.tb1 = self.W1.copy(), self.b1.copy()
        self.tW2, self.tb2 = self.W2.copy(), self.b2.copy()

    # --- persistence ---
    def save(self, path: str) -> None:
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path: str) -> "MLPQAgent":
        d = np.load(path)
        agent = cls(hidden=d["b1"].shape[0])
        agent.W1, agent.b1, agent.W2, agent.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        agent._sync_target()
        agent.eps = 0.0
        return agent
