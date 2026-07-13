"""Observation encoding, action space, and a linear Q-learning agent for MiniClash.

Why not the tabular Q-learning from Gridworld.py? MiniClash's state is continuous and
unbounded, so there's no finite table to index. We keep the Q-learning *idea* but swap
the table for a function approximator over a small, hand-crafted feature vector.
`LinearQAgent` here is the simple baseline; `mlp_agent.MLPQAgent` is the upgrade that
can actually learn to hold elixir and counterpush (see that file and train.py).

Two design choices keep the problem tractable:
  * Observations are encoded from the acting team's *own* perspective (progress is
    measured "toward the enemy"), so one agent design works for either side.
  * The raw action space -- (hand slot) x (spawn_x) x (spawn_y) -- is ~1000 placements
    per tick. We collapse it to 17 discrete choices: no-op, or (slot x lane x depth).
    Spells auto-target the densest enemy cluster, so the agent only decides *when*.
"""

import numpy as np

from ClashSim import (
    MiniClash, P0, P1, W_TILES, H_TILES,
    RIVER_Y0, RIVER_Y1, LEFT_LANE_CENTER_X, RIGHT_LANE_CENTER_X,
)

# ====================
# Action space
# ====================
# 0 = no-op. Others index a (hand slot, lane, depth) triple.
N_SLOTS = 4
N_LANES = 2
N_DEPTHS = 2  # 0 = back (build up), 1 = bridge (push)
N_ACTIONS = 1 + N_SLOTS * N_LANES * N_DEPTHS  # 17

# Concrete deploy rows per team. Chosen to be legal tiles (own side, not river/tower).
_DEPTH_Y = {
    P0: {0: 8, 1: RIVER_Y0 - 1},   # top team pushes "down" toward larger y
    P1: {0: H_TILES - 8, 1: RIVER_Y1 + 1},
}
_LANE_X = {0: LEFT_LANE_CENTER_X, 1: RIGHT_LANE_CENTER_X}


def _decode(action: int) -> tuple[int, int, int]:
    """action index (1..16) -> (slot, lane, depth)."""
    a = action - 1
    slot = a // (N_LANES * N_DEPTHS)
    rem = a % (N_LANES * N_DEPTHS)
    lane = rem // N_DEPTHS
    depth = rem % N_DEPTHS
    return slot, lane, depth


def _spell_target(env: MiniClash, team: int, radius: float) -> tuple[int, int]:
    """Aim a spell at the enemy unit with the most neighbours within `radius`.
    Falls back to chipping the nearest standing enemy crown tower if no units exist."""
    enemy = P1 if team == P0 else P0
    foes = [u for u in env.state.units.values() if u.team == enemy and u.hp > 0]
    if foes:
        best, best_n = foes[0], -1
        for c in foes:
            n = sum(1 for u in foes if (u.x - c.x) ** 2 + (u.y - c.y) ** 2 <= radius ** 2)
            if n > best_n:
                best, best_n = c, n
        return int(round(best.x)), int(round(best.y))
    # No troops on the field: chip a crown tower instead of wasting the spell.
    crowns = [t for t in env.state.towers if t.team == enemy and t.kind != "king" and t.hp > 0]
    tw = crowns[0] if crowns else next(t for t in env.state.towers if t.team == enemy and t.kind == "king")
    cx, cy = tw.center
    return int(cx), int(cy)


def action_to_env(env: MiniClash, team: int, action: int) -> tuple[int, int, int] | None:
    """Map a discrete action to the engine's (hand_slot, spawn_x, spawn_y) or None.
    For spell cards the lane/depth are ignored and the target is chosen automatically."""
    if action == 0:
        return None
    slot, lane, depth = _decode(action)
    cdef = env.card_defs[env.state.players[team].hand[slot]]
    if cdef.is_spell:
        tx, ty = _spell_target(env, team, cdef.spell_radius)
        return slot, tx, ty
    return slot, _LANE_X[lane], _DEPTH_Y[team][depth]


def legal_mask(env: MiniClash, team: int) -> np.ndarray:
    """Boolean mask of playable actions. No-op is always legal; a card action is legal
    only if the team can afford the card currently in that hand slot."""
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[0] = True
    ps = env.state.players[team]
    for a in range(1, N_ACTIONS):
        slot, _, _ = _decode(a)
        if ps.elixir >= env.card_defs[ps.hand[slot]].cost:
            mask[a] = True
    return mask


# ====================
# Observation encoder
# ====================
N_FEATURES = 41


def _progress(team: int, y: float) -> float:
    """0.0 at a team's own back line, 1.0 at the enemy's back line."""
    frac = y / H_TILES
    return frac if team == P0 else (1.0 - frac)


def _tower_frac(env: MiniClash, team: int, kind: str) -> float:
    for t in env.state.towers:
        if t.team == team and t.kind == kind:
            return max(0.0, t.hp) / t.max_hp
    return 0.0


def encode(env: MiniClash, team: int) -> np.ndarray:
    """Perspective-normalized feature vector for `team`. First entry is a bias term."""
    st = env.state
    enemy = P1 if team == P0 else P0
    f = np.zeros(N_FEATURES, dtype=np.float32)

    f[0] = 1.0  # bias
    f[1] = st.players[team].elixir / 10.0
    f[2] = st.players[enemy].elixir / 10.0
    f[3] = (st.players[team].elixir - st.players[enemy].elixir) / 10.0
    f[4] = st.time_left / 180.0

    # Committed elixir on the field, and pressure on each half.
    on_my_half = (lambda y: y < RIVER_Y0) if team == P0 else (lambda y: y > RIVER_Y1)
    on_enemy_half = (lambda y: y > RIVER_Y1) if team == P0 else (lambda y: y < RIVER_Y0)
    my_value = en_value = 0.0
    en_on_mine = my_on_enemy = 0
    my_cnt = [0, 0]; en_cnt = [0, 0]
    my_adv = [0.0, 0.0]; en_adv = [0.0, 0.0]
    en_tank = [0.0, 0.0]  # is an enemy building-targeter (Giant) pushing this lane?
    for u in st.units.values():
        if u.hp <= 0:
            continue
        adv = _progress(u.team, u.y)
        if u.team == team:
            my_value += u.udef.cost
            my_cnt[u.lane] += 1
            my_adv[u.lane] = max(my_adv[u.lane], adv)
            if on_enemy_half(u.y):
                my_on_enemy += 1
        else:
            en_value += u.udef.cost
            en_cnt[u.lane] += 1
            en_adv[u.lane] = max(en_adv[u.lane], adv)
            if u.udef.targets_buildings:
                en_tank[u.lane] = 1.0
            if on_my_half(u.y):
                en_on_mine += 1

    f[5] = my_value / 20.0
    f[6] = en_value / 20.0
    f[7] = min(en_on_mine, 5) / 5.0
    f[8] = min(my_on_enemy, 5) / 5.0

    i = 9
    for lane in (0, 1):
        f[i] = min(my_cnt[lane], 5) / 5.0; i += 1
        f[i] = min(en_cnt[lane], 5) / 5.0; i += 1
        f[i] = my_adv[lane]; i += 1
        f[i] = en_adv[lane]; i += 1
        f[i] = en_tank[lane]; i += 1

    for t in (team, enemy):
        for kind in ("crownL", "crownR", "king"):
            f[i] = _tower_frac(env, t, kind); i += 1

    ps = st.players[team]
    for slot in range(N_SLOTS):
        cdef = env.card_defs[ps.hand[slot]]
        f[i] = 1.0 if ps.elixir >= cdef.cost else 0.0; i += 1
        f[i] = 1.0 if cdef.is_spell else 0.0; i += 1
        f[i] = 1.0 if cdef.targets_buildings else 0.0; i += 1
        f[i] = cdef.cost / 10.0; i += 1

    assert i == N_FEATURES, f"feature count mismatch: {i} != {N_FEATURES}"
    return f


# ====================
# Linear Q-learning agent (simple baseline)
# ====================
class LinearQAgent:
    """Q(s, a) = W[a] . phi(s), trained with epsilon-greedy TD(0) updates."""

    def __init__(self, lr: float = 0.02, gamma: float = 0.99,
                 eps: float = 1.0, td_clip: float = 1.0, seed: int | None = None):
        self.W = np.zeros((N_ACTIONS, N_FEATURES), dtype=np.float32)
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.td_clip = td_clip
        self.rng = np.random.default_rng(seed)

    def q_values(self, phi: np.ndarray) -> np.ndarray:
        return self.W @ phi

    def select_action(self, phi: np.ndarray, mask: np.ndarray, greedy: bool = False) -> int:
        legal = np.flatnonzero(mask)
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.choice(legal))
        q = np.where(mask, self.q_values(phi), -np.inf)
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))  # random tie-break

    def update(self, phi: np.ndarray, action: int, reward: float,
               phi_next: np.ndarray, mask_next: np.ndarray, done: bool) -> None:
        target = reward
        if not done:
            q_next = np.where(mask_next, self.q_values(phi_next), -np.inf)
            target += self.gamma * float(q_next.max())
        td = target - float(self.W[action] @ phi)
        td = float(np.clip(td, -self.td_clip, self.td_clip))
        self.W[action] += self.lr * td * phi

    def save(self, path: str) -> None:
        np.savez(path, W=self.W)

    @classmethod
    def load(cls, path: str) -> "LinearQAgent":
        agent = cls(eps=0.0)
        agent.W = np.load(path)["W"]
        return agent
