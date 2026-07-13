# MiniClash — Self-Play Reinforcement Learning

A Clash Royale–style battle simulator where two bots learn to play
**against each other** from scratch via self-play reinforcement learning.

Two agents share no weights and improve simultaneously, so each one's opponent keeps
getting stronger as training proceeds — the defining feature (and challenge) of
self-play. After a few minutes of training on a laptop CPU, both learned agents beat
the built-in random bot in over 80% of games from either side.

## What's here

| File | Purpose |
|------|---------|
| `ClashSim.py` | The game engine + a pygame renderer. Two lanes, a river with bridges, crown/king towers, elixir, a deck/hand cycle, and two unit types (Knight, Archer). Exposes a Gym-style RL API: `reset()` and `step(actions) -> (state, rewards, done)`. |
| `agent.py` | The observation encoder, discrete action space, and a linear function-approximation Q-learning agent. |
| `train.py` | Self-play training loop, greedy evaluation, and a `--watch` mode to render trained agents. |
| `Gridworld.py` | A standalone tabular Q-learning warm-up on a 5×5 grid (kept as a reference). |

## Quick start

```bash
# create/activate a venv, then:
pip install numpy pygame

python train.py                     # train both bots, evaluate, save weights
python train.py --episodes 2000     # train longer
python train.py --watch             # watch the trained bots play (space = pause, r = reset)
python ClashSim.py                  # original random-bot demo
```

Training saves two weight files: `crsim_agents_p0.npz` and `crsim_agents_p1.npz`.

## How the learning works

MiniClash's state is continuous and unbounded (float unit positions, a variable number
of units, elixir, tower HPs), so the tabular Q-learning in `Gridworld.py` has no finite
table to index. We keep the Q-learning *idea* but replace the table with a **linear
function approximator**:

```
Q(s, a) = W[a] · φ(s)
```

- **Observation `φ(s)`** — a 26-feature vector, normalized and encoded from the acting
  team's *own* perspective (progress is measured "toward the enemy"), so one agent design
  works for either side. Features cover elixir, per-lane unit pressure, tower HP
  fractions, and the current hand.
- **Actions** — the raw action space (hand slot × spawn_x × spawn_y ≈ 1000 placements per
  tick) is collapsed to **17 discrete choices**: no-op, or (slot × lane × depth), with
  legal-action masking so a card is only playable when it can be afforded.
- **Reward** — dense shaping from the tower-HP swing each tick (you gain by damaging enemy
  towers, lose when yours take damage), plus a terminal ±5 for winning/losing.
- **Training** — ε-greedy TD(0) updates, ε decaying 1.0 → 0.05, both agents learning online
  from their own experience each tick.

## Results

800 episodes (~3 min on CPU). Self-play stays balanced (~40–60% each, no draws), and both
learned agents dominate the random baseline:

| Match | Wins |
|-------|------|
| trained P0 vs trained P1 | 194 – 106 |
| trained P0 vs **random** P1 | **278 – 22** |
| **random** P0 vs trained P1 | 48 – **250** |
