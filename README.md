# MiniClash — Self-Play Reinforcement Learning

A Clash Royale–style battle simulator where two bots learn to play **against each other**
from scratch via self-play reinforcement learning.

Two agents improve simultaneously, so each one's opponent keeps getting stronger as
training proceeds — the defining feature (and challenge) of self-play. Three learners are
implemented, from a tabular-style linear baseline up to **PPO**. The PPO agent learns to
*hold elixir* and commit a coordinated push — the strategic behaviour the value-based
agents never discovered — and roughly doubles the win rate against a strong hand-written
heuristic.

## What's here

| File | Purpose |
|------|---------|
| `ClashSim.py` | Game engine + pygame renderer. Two lanes, a river with bridges, crown/king towers, an elixir economy, a deck/hand cycle, five cards, and a Gym-style RL API: `reset()` and `step(actions) -> (state, rewards, done)`. Also holds `random_bot` and the stronger `heuristic_bot`. |
| `agent.py` | Observation encoder, the 17-action discrete action space, and `LinearQAgent` (simple baseline). |
| `mlp_agent.py` | `MLPQAgent` — value-based upgrade: a NumPy MLP Q-function with a replay buffer, target network, and n-step returns. |
| `ppo_agent.py` | `PPOAgent` — on-policy clipped actor-critic (shared policy, GAE, entropy bonus), all NumPy with a numeric gradient check. |
| `train.py` | Self-play training for all three learners, evaluation vs the heuristic/random bots, and a `--watch` renderer. |
| `Gridworld.py` | A standalone tabular Q-learning warm-up on a 5×5 grid (kept as a reference). |

## Quick start

```bash
pip install numpy pygame

python train.py --agent ppo                 # train the PPO self-play policy, evaluate, save
python train.py --agent ppo --episodes 4000 # train longer
python train.py --agent ppo --watch         # watch the trained policy play (space = pause, r = reset)
python train.py --agent mlp                 # the value-based learner
python train.py --agent linear              # the simplest baseline learner
python ClashSim.py                          # heuristic-vs-heuristic demo (no learning)
python ppo_agent.py                         # numerically verify the PPO backprop
```

Training saves two weight files, `<model>_p0.npz` and `<model>_p1.npz` (PPO shares one
policy across both).

## The game

- **Troops** — Knight (sturdy melee), Archer (ranged), Giant (slow tank that *ignores
  troops and marches on towers*), Goblins (a fast, fragile 3-unit swarm).
- **Spell** — Arrows (instant area damage; wipes swarms, chips towers). Cast anywhere.
- **Opponents** — `random_bot` (deploys at random) and `heuristic_bot`, a rule-based
  baseline that defends the threatened lane, answers swarms with Arrows, and pushes with
  a Giant when elixir is plentiful.

## How the learning works

MiniClash's state is continuous and unbounded, so the tabular Q-learning in
`Gridworld.py` has no finite table to index. We keep the Q-learning *idea* but replace
the table with a function approximator over a 41-feature, perspective-normalized
observation (elixir, committed field value, per-lane pressure, tower HP, hand contents).
The raw action space (~1000 placements/tick) is collapsed to **17 discrete choices**:
no-op, or (slot × lane × depth); spells auto-target the densest enemy cluster, so the
agent only decides *when* to cast.

**Reward** is dense: net tower damage each tick, a net **elixir-trade** term (killing more
elixir value than you lose in a fight), small spend/overflow penalties to discourage
mindless dumping, plus a terminal ±10 for the result.

Three learners are provided, in increasing order of capability:

- **`LinearQAgent`** — `Q(s,a) = W[a]·φ(s)`, online one-step TD. A solid baseline.
- **`MLPQAgent`** — value-based upgrade: a ReLU-MLP Q-function with **n-step returns**
  (delayed credit for a counterpush), a **replay buffer**, and a **target network**.
- **`PPOAgent`** — on-policy clipped actor-critic. A single shared policy plays both sides
  (the observation is perspective-normalized), trained with **GAE**, a clipped surrogate,
  and an annealed **entropy bonus**. All NumPy with hand-written backprop + Adam; run
  `python ppo_agent.py` to numerically verify the gradients.

### Why PPO — and what it learned

Both Q-learning agents get stuck in an **elixir-dumping local optimum**: they spend on a
cheap troop the instant they can afford one (average elixir ~1.4/10), never commit the
Giant, and get overwhelmed. The counterpush machinery (n-step returns, elixir-trade
reward, spend penalties) is all present — but the fix isn't reward-weakness. A controlled
experiment cranking `SPEND_SCALE` 5× *did not change the behaviour at all*. The bottleneck
is **exploration**: the "hold to 5+, then Giant + support" line needs ~15 coordinated
decisions in a row, which epsilon-greedy (random single-action jitter) essentially never
samples, so the payoff of holding is never experienced.

PPO explores with a *stochastic policy* shaped by an entropy bonus, which samples coherent
multi-step behaviour far more readily. The result: it **learns to hold elixir** (average
rises to ~2.6/10 in play, and its greedy trace holds all the way to ~7) and commits the
Giant far more often — the strategic behaviour the value-based agents never found.

> **Evaluation note:** a stochastic policy must be evaluated by *sampling*, not greedy
> argmax. PPO's argmax over-selects no-op (it has plurality but not majority probability),
> collapsing the agent into passivity — greedy eval reports ~8% vs the heuristic while the
> same policy sampled scores ~45-50%. `evaluate()`/`--watch` sample automatically for PPO.

## Results

Win rate vs each baseline, ~400 games, averaged over both sides (PPO sampled; value-based
agents greedy):

| Agent | vs `random_bot` | vs `heuristic_bot` |
|-------|-----------------|--------------------|
| `LinearQAgent` / `MLPQAgent` (self-play) | ~85% | ~15–21%            |
| **`PPOAgent` (self-play)** | **~90%** | **~45-50%**        |

PPO more than doubles the win rate against the strong heuristic and, more importantly,
exhibits the target behaviour: holding elixir for a coordinated push instead of dumping.

## Development journey

This project was built iteratively, and most of the learning came from diagnosing why
things *didn't* work. The process:

1. **Tabular Q-learning warm-up** (`Gridworld.py`) to get the RL loop right on a problem
   with a finite state table.

2. **Built the MiniClash engine**, then found it couldn't produce a decision at all: every
   game ended in a draw. Instrumentation showed units got stuck ~2.5–3 tiles from the enemy
   king (out of attack range) because a "bridge-crossing" movement guard fired everywhere,
   not just at the river. Rewrote movement as a waypoint scheme → games became decisive and
   symmetric. (Also fixed a winner off-by-one and a broken HP-bar draw.)

3. **Wired up the RL reward** — `step()` had returned a hardcoded `0.0`, so nothing could
   learn. Added dense tower-damage shaping plus a terminal win/loss bonus.

4. **Linear Q-learning self-play** worked as a proof of life (beat random handily), so I
   built `heuristic_bot` as a *much* stronger yardstick and added real cards (Giant,
   Goblins, Arrows) to create genuine elixir-trade decisions.

5. **MLP Q-learning** (n-step returns, replay, target net) + an elixir-trade reward +
   opponent-pool anchoring against the heuristic. Beat random ~85% — but stalled at
   ~15–21% vs the heuristic across several runs.

6. **Diagnosed the plateau.** Instrumenting games revealed an *elixir-dumping local
   optimum*: average elixir sat at ~1.4/10 and the Giant was almost never played. The agent
   was spending on cheap troops the instant it could afford them.

7. **Ran a controlled experiment** to find the cause: cranked the spend penalty 5×,
   changing one variable. Behaviour did **not** move at all. That ruled out "reward too
   weak" and pointed at **exploration** — epsilon-greedy never samples the ~15-decision
   "hold, then commit a push" sequence, so holding never looks rewarding.

8. **Implemented PPO from scratch** (gradient-checked before training). Its entropy-driven
   stochastic exploration finally discovered holding elixir, ~doubling the win rate vs the
   heuristic.

9. **Caught an evaluation pitfall:** PPO's greedy argmax over-picks no-op and looks
   passive (~8% vs heuristic), while the *same policy sampled* scores ~45-50%. Fixed the
   evaluator to sample for stochastic policies.

The takeaway that generalizes: **measure behavior, not just win rate, and change one
variable at a time** — the spend-penalty experiment is what turned "the agent is bad" into
the specific, correct diagnosis "this is an exploration problem."
