"""Self-play training for MiniClash.

Two agents (one per team) play each other and learn simultaneously, so each one's
opponent is non-stationary -- the defining feature of self-play. Choose the learner with
--agent:
    linear : LinearQAgent, online one-step TD (simple baseline)
    mlp    : MLPQAgent, n-step returns + replay + target net (value-based)
    ppo    : PPOAgent, on-policy clipped actor-critic with a shared policy that plays
             both sides. Explores coherent multi-step strategies (entropy-driven), which
             is what the value-based agents couldn't -- see ppo_agent.py.

Usage:
    python train.py --agent ppo                 # train, save, eval vs the heuristic bot
    python train.py --agent ppo --episodes 3000
    python train.py --agent ppo --watch         # render the saved agents playing
"""

import argparse
from collections import deque

import numpy as np

from ClashSim import MiniClash, P0, P1, heuristic_bot, random_bot
from agent import LinearQAgent, encode, legal_mask, action_to_env
from mlp_agent import MLPQAgent
from ppo_agent import PPOAgent, compute_gae


# ====================
# n-step return helper (MLP path)
# ====================
class NStepBuffer:
    """Turns a per-team stream of (phi, action, reward) into n-step transitions and
    files them in the agent's replay buffer. Bootstraps with gamma**n except when the
    episode ends inside the window (then no future value is added)."""

    def __init__(self, agent: MLPQAgent, n: int, gamma: float):
        self.agent, self.n, self.gamma = agent, n, gamma
        self.q: deque = deque()

    def push(self, phi, action, reward, phi_next, mask_next, done):
        self.q.append((phi, action, reward))
        if len(self.q) >= self.n:
            self._emit(phi_next, mask_next, self.gamma ** self.n)
        if done:
            while self.q:
                self._emit(phi_next, mask_next, 0.0)

    def _emit(self, phi_n, mask_n, bootstrap):
        G, g = 0.0, 1.0
        for (_, _, r) in self.q:
            G += g * r
            g *= self.gamma
        phi0, a0, _ = self.q.popleft()
        self.agent.remember(phi0, a0, G, phi_n, bootstrap, mask_n)


# ====================
# Episodes
# ====================
def episode_mlp(env, agents, nbuf, max_ticks, opp_team=None, opp_bot=None) -> int | None:
    """Play one MLP training game. Teams other than `opp_team` are learning agents that
    store n-step transitions and take gradient steps; `opp_team` (if set) is driven by a
    scripted `opp_bot` and does not learn. With opp_team=None this is pure self-play;
    with an opponent it anchors a learner against a fixed strategy (opponent-pool / league
    training), which stops self-play from drifting into a mutually-exploitable policy."""
    learners = [t for t in (P0, P1) if t != opp_team]
    env.reset()
    done, ticks = False, 0
    while not done and ticks < max_ticks:
        phi = {t: encode(env, t) for t in learners}
        mask = {t: legal_mask(env, t) for t in learners}
        acts = {t: agents[t].select_action(phi[t], mask[t]) for t in learners}
        env_actions = {t: action_to_env(env, t, acts[t]) for t in learners}
        if opp_team is not None:
            env_actions[opp_team] = opp_bot(env, opp_team)
        _, rewards, done = env.step(env_actions)
        ticks += 1
        for t in learners:
            phi_n = encode(env, t)
            mask_n = legal_mask(env, t)
            nbuf[t].push(phi[t], acts[t], rewards[t], phi_n, mask_n, done)
            if ticks % 2 == 0:  # one gradient step every other tick is plenty
                agents[t].learn()
    return env.state.winner


def selfplay_episode_linear(env, agents, max_ticks) -> int | None:
    env.reset()
    done, ticks = False, 0
    while not done and ticks < max_ticks:
        phi = {t: encode(env, t) for t in (P0, P1)}
        mask = {t: legal_mask(env, t) for t in (P0, P1)}
        acts = {t: agents[t].select_action(phi[t], mask[t]) for t in (P0, P1)}
        env_actions = {t: action_to_env(env, t, acts[t]) for t in (P0, P1)}
        _, rewards, done = env.step(env_actions)
        ticks += 1
        phi_n = {t: encode(env, t) for t in (P0, P1)}
        mask_n = {t: legal_mask(env, t) for t in (P0, P1)}
        for t in (P0, P1):
            agents[t].update(phi[t], acts[t], rewards[t], phi_n[t], mask_n[t], done)
    return env.state.winner


def play_vs_bot(env, agent, team, bot, max_ticks, greedy=True) -> int | None:
    """One game: `agent` controls `team`, `bot` controls the other side. Value-based
    agents evaluate greedily; a stochastic policy (PPO) must be *sampled* -- its greedy
    argmax over-selects no-op (plurality but not majority), collapsing into passivity."""
    other = P1 if team == P0 else P0
    env.reset()
    done, ticks = False, 0
    while not done and ticks < max_ticks:
        a_agent = action_to_env(env, team, agent.select_action(encode(env, team), legal_mask(env, team), greedy=greedy))
        acts = {team: a_agent, other: bot(env, other)}
        _, _, done = env.step(acts)
        ticks += 1
    return env.state.winner


# ====================
# PPO (on-policy)
# ====================
def collect_ppo_episode(env, agent, traj_buf, max_ticks, opp_team=None, opp_bot=None) -> int | None:
    """Play one episode with the shared PPO policy and append each learner's trajectory
    (obs/mask/action/logprob/value/reward + a bootstrap value) to `traj_buf`. With
    opp_team set, that side is a scripted bot and only the learner side is recorded."""
    learners = [t for t in (P0, P1) if t != opp_team]
    env.reset()
    data = {t: {k: [] for k in ("phi", "mask", "act", "logp", "val", "rew")} for t in learners}
    done, ticks = False, 0
    while not done and ticks < max_ticks:
        phi = {t: encode(env, t) for t in learners}
        mask = {t: legal_mask(env, t) for t in learners}
        chosen, env_actions = {}, {}
        for t in learners:
            a, lp, v = agent.act_collect(phi[t], mask[t])
            chosen[t] = (a, lp, v)
            env_actions[t] = action_to_env(env, t, a)
        if opp_team is not None:
            env_actions[opp_team] = opp_bot(env, opp_team)
        _, rewards, done = env.step(env_actions)
        ticks += 1
        for t in learners:
            a, lp, v = chosen[t]
            d = data[t]
            d["phi"].append(phi[t]); d["mask"].append(mask[t]); d["act"].append(a)
            d["logp"].append(lp); d["val"].append(v); d["rew"].append(rewards[t])

    # Bootstrap: 0 for a real terminal (someone won), else V(final obs) for a timeout.
    terminal = env.state.winner is not None
    for t in learners:
        boot = 0.0 if terminal else agent.policy(encode(env, t), legal_mask(env, t))[1]
        traj_buf.append((data[t], boot))
    return env.state.winner


def _buf_steps(traj_buf) -> int:
    return sum(len(d["rew"]) for d, _ in traj_buf)


def _flatten_rollout(traj_buf, gamma, lam):
    Xs, Ms, As, LPs, ADVs, RETs = [], [], [], [], [], []
    for d, boot in traj_buf:
        adv, ret = compute_gae(d["rew"], d["val"], boot, gamma, lam)
        Xs.append(np.asarray(d["phi"], np.float32))
        Ms.append(np.asarray(d["mask"], bool))
        As.append(np.asarray(d["act"]))
        LPs.append(np.asarray(d["logp"], np.float32))
        ADVs.append(adv); RETs.append(ret)
    return (np.concatenate(Xs), np.concatenate(Ms), np.concatenate(As),
            np.concatenate(LPs), np.concatenate(ADVs), np.concatenate(RETs))


def train_ppo(episodes, max_ticks, gamma, lam, out_path, p_heuristic=0.4,
              steps_per_update=8192):
    env = MiniClash()
    agent = PPOAgent(seed=0)
    rng = np.random.default_rng(123)

    traj_buf = []
    window = []
    heur_games = heur_wins = 0
    updates = 0
    last_info = {}

    for ep in range(episodes):
        # Anneal the entropy bonus: explore coherently early, sharpen the policy late.
        agent.c_entropy = 0.02 + (0.004 - 0.02) * (ep / max(1, episodes - 1))

        if rng.random() < p_heuristic:
            learner = int(rng.integers(2))
            opp = P1 if learner == P0 else P0
            w = collect_ppo_episode(env, agent, traj_buf, max_ticks, opp_team=opp, opp_bot=heuristic_bot)
            heur_games += 1
            heur_wins += (w == learner)
        else:
            w = collect_ppo_episode(env, agent, traj_buf, max_ticks)
            window.append(w)

        if _buf_steps(traj_buf) >= steps_per_update:
            last_info = agent.update(*_flatten_rollout(traj_buf, gamma, lam), epochs=4, minibatch=256)
            traj_buf = []
            updates += 1

        report_every = max(1, episodes // 20)
        if (ep + 1) % report_every == 0:
            n = max(1, len(window))
            w0, w1 = window.count(P0), window.count(P1)
            hr = f" | vs-heur {heur_wins}/{heur_games}" if heur_games else ""
            li = f" | H_dec={last_info.get('entropy_dec', 0):.2f} vloss={last_info.get('value', 0):.3f}" if last_info else ""
            print(f"ep {ep + 1:5d}/{episodes} | upd {updates:3d} | self-play last {n}: "
                  f"P0 {w0/n:4.0%} P1 {w1/n:4.0%}{hr}{li}")
            window = []
            heur_games = heur_wins = 0

    # Shared policy: save the same weights to both slots so eval/watch load unchanged.
    agent.save(out_path.replace(".npz", "_p0.npz"))
    agent.save(out_path.replace(".npz", "_p1.npz"))
    print(f"Saved shared PPO policy to {out_path.replace('.npz', '_p0.npz')} / _p1.npz")
    return {P0: agent, P1: agent}


# ====================
# Train / evaluate
# ====================
def make_agents(kind: str):
    if kind == "mlp":
        return {P0: MLPQAgent(seed=0), P1: MLPQAgent(seed=1)}
    return {P0: LinearQAgent(seed=0), P1: LinearQAgent(seed=1)}


def train(kind, episodes, max_ticks, gamma, n_step, out_path, p_heuristic=0.5):
    env = MiniClash()
    agents = make_agents(kind)
    nbuf = {t: NStepBuffer(agents[t], n_step, gamma) for t in (P0, P1)} if kind == "mlp" else None
    rng = np.random.default_rng(123)

    eps_start, eps_end = 1.0, 0.05
    window = []
    heur_games = heur_wins = 0  # rolling record of learner-vs-heuristic anchor games

    for ep in range(episodes):
        eps = eps_start + (eps_end - eps_start) * (ep / max(1, episodes - 1))
        for a in agents.values():
            a.eps = eps

        # A fraction of MLP games anchor one (randomly chosen) learner against the
        # heuristic; the rest are pure self-play. Linear stays pure self-play.
        anchor = kind == "mlp" and rng.random() < p_heuristic
        if anchor:
            learner = int(rng.integers(2))
            opp = P1 if learner == P0 else P0
            winner = episode_mlp(env, agents, nbuf, max_ticks, opp_team=opp, opp_bot=heuristic_bot)
            heur_games += 1
            heur_wins += (winner == learner)
        elif kind == "mlp":
            winner = episode_mlp(env, agents, nbuf, max_ticks)
            window.append(winner)
        else:
            winner = selfplay_episode_linear(env, agents, max_ticks)
            window.append(winner)

        report_every = max(1, episodes // 20)
        if (ep + 1) % report_every == 0:
            n = max(1, len(window))
            w0, w1 = window.count(P0), window.count(P1)
            hr = f"  vs-heur {heur_wins}/{heur_games}" if heur_games else ""
            print(f"ep {ep + 1:5d}/{episodes} | eps={eps:.2f} | self-play last {n}: "
                  f"P0 {w0/n:4.0%} P1 {w1/n:4.0%}{hr}")
            window = []
            heur_games = heur_wins = 0

    agents[P0].save(out_path.replace(".npz", "_p0.npz"))
    agents[P1].save(out_path.replace(".npz", "_p1.npz"))
    print(f"Saved agents to {out_path.replace('.npz', '_p0.npz')} / _p1.npz")
    return agents


def evaluate(agents, games, max_ticks):
    env = MiniClash()
    for a in agents.values():
        a.eps = 0.0
    greedy = not isinstance(agents[P0], PPOAgent)  # PPO is stochastic: sample, don't argmax

    def bench(bot, name):
        won = 0
        for _ in range(games):
            won += (play_vs_bot(env, agents[P0], P0, bot, max_ticks, greedy) == P0)
        for _ in range(games):
            won += (play_vs_bot(env, agents[P1], P1, bot, max_ticks, greedy) == P1)
        print(f"trained vs {name:9s}: won {won}/{2*games} ({won/(2*games):.1%}) from both sides")

    bench(heuristic_bot, "heuristic")
    bench(random_bot, "random")


def watch(kind, model_path, max_ticks):
    import pygame
    from ClashSim import Renderer
    Loader = {"mlp": MLPQAgent, "ppo": PPOAgent}.get(kind, LinearQAgent)
    agents = {
        P0: Loader.load(model_path.replace(".npz", "_p0.npz")),
        P1: Loader.load(model_path.replace(".npz", "_p1.npz")),
    }
    for a in agents.values():
        a.eps = 0.0
    greedy = kind != "ppo"  # PPO is stochastic: sample so it actually plays cards

    env = MiniClash()
    ren = Renderer()
    running, paused, ticks = True, False, 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    env.reset(); ticks = 0
        if not paused and env.state.winner is None and ticks < max_ticks:
            env_actions = {t: action_to_env(env, t, agents[t].select_action(
                encode(env, t), legal_mask(env, t), greedy=greedy)) for t in (P0, P1)}
            env.step(env_actions)
            ticks += 1
        ren.draw(env.state)
        ren.tick()
    ren.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["linear", "mlp", "ppo"], default="ppo")
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--max-ticks", type=int, default=2000)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--lam", type=float, default=0.95, help="GAE lambda (ppo)")
    ap.add_argument("--n-step", type=int, default=40, help="n-step returns (mlp)")
    ap.add_argument("--model", default="crsim_agents.npz")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--eval", type=int, default=100)
    args = ap.parse_args()

    if args.watch:
        watch(args.agent, args.model, args.max_ticks)
        return

    if args.agent == "ppo":
        agents = train_ppo(args.episodes, args.max_ticks, args.gamma, args.lam, args.model)
    else:
        agents = train(args.agent, args.episodes, args.max_ticks, args.gamma, args.n_step, args.model)
    if args.eval > 0:
        evaluate(agents, args.eval, args.max_ticks)


if __name__ == "__main__":
    main()
